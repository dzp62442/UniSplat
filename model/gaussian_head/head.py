from typing import List, Dict, Tuple, Union

import utils3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from model.loss_func.percept_loss import LPIPS
from ..layers.patch_embed import PatchEmbed
from ..layers.gaussian_dyn import GaussianRenderer_dyn
from ..layers.spconv_unet import get_voxel_centers, project_world_points_to_images
from ..layers.vision_transformer import vit_small
from .utils import create_uv_grid, position_grid_to_embed, is_point_in_frustum_batch, \
    align_points_scale_z_shift, mask_aware_nearest_resize, normalize_intrinsics
from .unet import UNet
from .head_layers import _make_scratch, _make_fusion_block_custom, custom_interpolate
from pi3.models.layers.transformer_head import TransformerDecoder
from pi3.models.layers.pos_embed import RoPE2D, PositionGetter


class GuassianHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        cfg: Dict = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        render_w, render_h = cfg.get('resolution')
        self.intermediate_layer_idx = [3, 8, 12, 16]

        self.norm = nn.LayerNorm(dim_in)
        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0,)
            for oc in out_channels])

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1),
            ])
        self.scratch = _make_scratch(out_channels, 256, expand=False,)
        # Attach additional modules to scratch.
        self.scratch.refinenet1 = _make_fusion_block_custom(256)
        self.scratch.refinenet2 = _make_fusion_block_custom(256)
        self.scratch.refinenet3 = _make_fusion_block_custom(256)
        self.scratch.refinenet4 = _make_fusion_block_custom(256, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.voxel_proj = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.plucker_to_embed = PatchEmbed(img_size=(render_h, render_w), patch_size=14, in_chans=6, embed_dim=1024, norm_layer=nn.LayerNorm)
        self.depth_embeds = PatchEmbed(img_size=(render_h, render_w), patch_size=14, in_chans=5, embed_dim=1024, norm_layer=nn.LayerNorm)
        self.embed_proj = nn.Sequential(
            nn.Linear(1024, 1024), nn.LayerNorm(1024),
            nn.GELU(), nn.Linear(1024, 2048))

        # gaussian layers
        self.to_gaussians = nn.Sequential(nn.Linear(320, 256), nn.GELU(), nn.Linear(256, 15),)
        self.to_gaussians_sky = nn.Sequential(nn.Linear(320, 256), nn.GELU(), nn.Linear(256, 15),)    
        self.feature_norm = nn.Identity()
        self.offset_act = lambda x: F.tanh(x) * cfg.get('offset_scale', 1.0)
        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: F.softplus(x, beta=2.0)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        self.renderer = GaussianRenderer_dyn(resolution = [render_h, render_w], znear = 0.1, zfar = 1000.0)
        self.cfg = cfg

        # loss
        self.perceptual_loss = LPIPS().eval() if cfg.get('build_perceptual_loss', True) else nn.Identity()
        self.loss_weight = cfg.get('Loss_weight', None)
        self.l1_loss_mask = cfg.get('l1_loss_mask', None)
        self.p_loss_mask = cfg.get('p_loss_mask', None)
        
        self.pts_range = torch.tensor(cfg.pts_range, dtype=torch.float)  # x_min, y_min, z_min, x_max, y_max, z_max
        self.voxel_size = torch.tensor(cfg.voxel_size, dtype=torch.float)  # voxel size in meters
        self.grid_size = (self.pts_range[3:6] - self.pts_range[:3]) / self.voxel_size[:3]  # grid size in voxels
        self.grid_size = self.grid_size.int()
        self.scale_xyz = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.scale_yz = self.grid_size[1] * self.grid_size[2]
        self.scale_z = self.grid_size[2]
        
        # UNet for Scaffold feature extraction
        self.unet = UNet(self.grid_size.numpy(), self.voxel_size.numpy(), self.pts_range.numpy(), cfg.voxel_gs_num)
        self.image_backbone = vit_small(img_size=cfg.get('dinov2_pretrain_img_size', render_w), patch_size=14, num_register_tokens=4, \
            interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0, init_values=1.0)
        self._resnet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self._resnet_mean = self._resnet_mean[None, None, :, None, None]
        self._resnet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self._resnet_std = self._resnet_std[None, None, :, None, None]
        self.dinov2_proj = nn.Sequential(nn.Linear(384, 2048), nn.LayerNorm(2048))
        self.history_queue = Gaussians_Queue_v2()
        
        # scale prediction
        self.scale_head = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1))
        self.shift_head = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1))
        self.rope = RoPE2D(freq=100.0)
        self.position_getter = PositionGetter()
        self.point_decoder = TransformerDecoder(
            in_dim=2048, dec_embed_dim=1024, dec_num_heads=16,
            out_dim=1024, rope=self.rope,)
   
    def forward(self, aggregated_tokens_list, images, patch_start_idx, input_dict_gs, output_dict_gs, \
                test=True, stage=3):
        if test:
            return self.test(aggregated_tokens_list, images, patch_start_idx, \
                input_dict_gs=input_dict_gs, output_dict_gs=output_dict_gs)
        return self._forward_impl(aggregated_tokens_list, images, patch_start_idx, \
            input_dict_gs=input_dict_gs, output_dict_gs=output_dict_gs, stage=stage)

    def test(self, aggregated_tokens_list, images, patch_start_idx, input_dict_gs, output_dict_gs):

        B, S, _, H, W = images.shape
        
        input_dict_gs['scene'] = aggregated_tokens_list['scene']
        input_dict_gs['frame'] = aggregated_tokens_list['frame']
        
        images_norm = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)
        images_norm = images_norm.view(B * S, 3, H, W)
        dinov2_features = self.image_backbone(images_norm)
        dinov2_features = dinov2_features['x_norm_patchtokens']
        dinov2_features = self.dinov2_proj(dinov2_features)

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # plucker_embedder
        device_id = images.device
        dtype = images.dtype
        rays_o = input_dict_gs["rays_o"].to(device_id, dtype)
        rays_d = input_dict_gs["rays_d"].to(device_id, dtype)
        pluckers = self.plucker_embedder(rays_o, rays_d)
        pluckers = pluckers.reshape(B*S, -1, H, W)
        plucker_embeds = self.plucker_to_embed(pluckers)
        
        ##### compute unscaled local depths #####
        local_points = aggregated_tokens_list['local_points']
        intrinsics = input_dict_gs["intrinsics"].clone()
        local_depth = local_points[..., 2]
        # pred scale and shift
        pos = self.position_getter(B * S, H//self.patch_size, W//self.patch_size, images.device)
        out = []
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list['intermidiate_output'][layer_idx][:, patch_start_idx:]
            out.append(x)
        out = torch.stack(out, dim=-1)
        out = out.mean(dim=-1)
        out_hidden = self.point_decoder(out, xpos=pos)
        out_hidden = out_hidden.mean(dim=1)
        pred_scale = self.scale_head(out_hidden).squeeze(-1).exp().reshape(B, S)
        pred_shift = self.shift_head(out_hidden).squeeze(-1).reshape(B, S)    
        depth_map_unscale = local_depth * pred_scale[:,:,None,None] + pred_shift[:,:,None, None]
        ##### compute unscaled local depths #####
        depth_conf_unscale = input_dict_gs['sky_mask']
        # give sky depth
        if self.cfg.get('sky_mask', 0.0) > 0.0:
            depth_map_unscale[depth_conf_unscale==0] = self.cfg['sky_mask']
        depth_map_unscale = torch.clamp(depth_map_unscale, max=self.cfg['sky_mask'], min=0.0)
        depthmap = torch.cat([depth_map_unscale[:, :, None, :, :]/self.cfg.get('sky_mask', 300.0),\
            depth_conf_unscale[:, :, None, :, :], images], dim=2)
        depthmap = depthmap.reshape(B * S, -1, H, W)
        depth_embeds = self.depth_embeds(depthmap)
        # aggregrate embeds
        agg_embeds = plucker_embeds + depth_embeds
        agg_embeds = self.embed_proj(agg_embeds)
        agg_embeds = agg_embeds.reshape(B*S, patch_h * patch_w, -1)

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list['intermidiate_output'][layer_idx][:, patch_start_idx:]
            x = x + agg_embeds + dinov2_features
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
        # FPN features
        fpn_out_project = self.voxel_proj(out[0])
        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        out = custom_interpolate(out,
            (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
            mode="bilinear", align_corners=True)
        img_guassian_features = self.feature_norm(out)
        means = rays_o + rays_d * depth_map_unscale[..., None]
        pixel_means = means.view(B, -1, 3)
        
        # each step select some view to render, reduce computation and memory
        out_view_num = output_dict_gs["rgb"].shape[1]
        select_mask = torch.zeros((out_view_num), dtype=torch.bool, device=device_id)
        select_mask[:] = True
        select_index = torch.nonzero(select_mask, as_tuple=False).squeeze(-1)
        # render parameters
        render_c2w = output_dict_gs["c2w"].to(device_id, dtype)
        render_fovxs = output_dict_gs["fovx"].to(device_id, dtype)
        render_fovys = output_dict_gs["fovy"].to(device_id, dtype)
        rgb_gt = output_dict_gs["rgb"].to(device_id, dtype)
        out_intrinsics = output_dict_gs["intrinsics"].to(device_id, dtype)

        # init voxel
        self.pts_range = self.pts_range.to(device=means.device)
        self.voxel_size = self.voxel_size.to(device=means.device)
        coords = []
        multimodel_gaussians = []
        for i in range(B):
            means = pixel_means[i]
            boundary = self.pts_range.clone()
            boundary[:3] = boundary[:3] + 0.01
            boundary[3:6] = boundary[3:6] - 0.01
            # use a more strict boundary to filter out points
            mask = (means[:, 0] > boundary[0]) & (means[:, 0] < boundary[3]) & \
                   (means[:, 1] > boundary[1]) & (means[:, 1] < boundary[4]) & \
                   (means[:, 2] > boundary[2]) & (means[:, 2] < boundary[5])
            count = mask.sum()
            batch_idx = means.new_zeros((count, 1)) + i
            batch_idx = batch_idx.int()
            gaussians_coords = means[mask]
            gaussians_coords = gaussians_coords - self.pts_range[None, :3]
            gaussians_coords = torch.floor(gaussians_coords / self.voxel_size[None, :3]).int()
            coords.append(torch.cat([batch_idx, gaussians_coords], dim=-1))
            rgbs = images[i].permute(0,2,3,1).reshape(-1, 3)
            multimodel_gaussians.append(torch.cat([means[mask], rgbs[mask]], dim=-1))
            
        coords = torch.cat(coords, dim=0)
        multimodel_gaussians = torch.cat(multimodel_gaussians, dim=0)
        merge_coords = coords[:, 0].int() * self.scale_xyz + coords[:, 1] * self.scale_yz + \
                       coords[:, 2] * self.scale_z + coords[:, 3]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)
        gaussians_voxel = torch_scatter.scatter_mean(multimodel_gaussians, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]] # bzyx
        voxel_centers = get_voxel_centers(voxel_coords[:,1:], 1, self.voxel_size, self.pts_range)
        # extract image_features
        intrinsics = input_dict_gs["intrinsics"]
        camera2lidar = input_dict_gs["camera2lidar"]
        gaussians_imgfeats = []
        for i in range(B):
            batch_mask = voxel_coords[:, 0] == i
            points = voxel_centers[batch_mask]
            projected_points, z_cam, valid_masks = project_world_points_to_images(points, intrinsics[i], camera2lidar[i], H, W)
            image_modality = fpn_out_project
            projected_points[:, :, 0] = (projected_points[:, :, 0] / W) * 2 - 1
            projected_points[:, :, 1] = (projected_points[:, :, 1] / H) * 2 - 1
            out_features = image_modality.new_zeros((image_modality.shape[0], projected_points.shape[1], image_modality.shape[1]))
            for view in range(valid_masks.shape[0]):
                input_features = image_modality[i*S+view:i*S+view+1]
                input_points = projected_points[view:view+1]
                input_mask = valid_masks[view:view+1]
                valid_points = input_points[input_mask]
                tmp_features = F.grid_sample(
                    input_features,
                    valid_points[None, None, :, :],
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )
                out_features[view][input_mask[0]] = tmp_features[0,:,0,:].transpose(0, 1)
            out_features = out_features.sum(dim=0) / (valid_masks.sum(dim=0)[:,None] + 1e-6)
            gaussians_imgfeats.append(out_features)
        gaussians_imgfeats = torch.cat(gaussians_imgfeats, dim=0)
        gaussians_voxel = torch.cat([gaussians_voxel, gaussians_imgfeats], dim=-1)
    
        # get history
        _, _, _, _, _, history_lidar2world, history_features, history_poses, voxel_opacities, \
            history_gaussians, history_gaussians_outview \
            = self.history_queue.get(B, aggregated_tokens_list['scene'], aggregated_tokens_list['frame'])
        current_lidar2world = aggregated_tokens_list['lidar2world']
        process_voxel_features, process_voxel_coords, process_voxel_pos, process_masks = [], [], [], []
        for i in range(B):
            if history_poses[i] is None:
                # no history, construct a dummy history
                tmp_history_features = torch.zeros((1, 64), dtype=gaussians_voxel.dtype, device=gaussians_voxel.device)
                tmp_history_poses = voxel_centers[voxel_coords[:,0] == i][:1]
                tmp_history_coords = tmp_history_poses - self.pts_range[None, :3]
                tmp_history_coords = torch.floor(tmp_history_coords / (self.voxel_size[None, :3] * 2)).int()
                bidx = tmp_history_coords.new_zeros((tmp_history_coords.shape[0], 1)) + i
                tmp_history_coords = torch.cat([bidx.int(), tmp_history_coords], dim=-1)
                tmp_mask = torch.tensor([False])
            else:
                history_2_current = (current_lidar2world[i].inverse() @ history_lidar2world[i])[0]
                ones = torch.ones((history_poses[i].shape[0], 1), dtype=images.dtype, device=images.device)
                history_poses_homo = torch.concat([history_poses[i], ones], dim=-1)
                history_poses_homo = (history_2_current @ history_poses_homo.T).T
                history_poses_homo = history_poses_homo[:,:-1]
                tmp_mask = (history_poses_homo[:, 0] > boundary[0]) & (history_poses_homo[:, 0] < boundary[3]) & \
                   (history_poses_homo[:, 1] > boundary[1]) & (history_poses_homo[:, 1] < boundary[4]) & \
                   (history_poses_homo[:, 2] > boundary[2]) & (history_poses_homo[:, 2] < boundary[5])
                
                tmp_history_poses = history_poses_homo[tmp_mask]
                tmp_history_features = history_features[i][tmp_mask]
                new_mask, _ = is_point_in_frustum_batch(tmp_history_poses, intrinsics[i], camera2lidar[i,:, :3,:3], camera2lidar[i,:, :3,3], 0.5, 72, 518, 350) 
                new_mask = new_mask.any(dim=-1)
                # in view filter
                inview_opacities = voxel_opacities[i][tmp_mask][new_mask]
                inview_features = tmp_history_features[new_mask]
                inview_poses = tmp_history_poses[new_mask]
                voxel_stream_quantile = min(self.cfg['voxel_stream_quantile'], self.cfg['voxel_stream_inview_num']/inview_opacities.shape[0])
                opacities_thre = torch.quantile(inview_opacities, 1 - voxel_stream_quantile)
                opacities_thre = max(opacities_thre, self.cfg.voxel_stream_opacity_thre)
                opacities_mask = inview_opacities > opacities_thre
                if opacities_mask.sum() == 0:
                    # at least keep one
                    opacities_mask[0] = True
                inview_features = inview_features[opacities_mask]
                inview_poses = inview_poses[opacities_mask]
                # out view filter
                if (~new_mask).sum() == 0:
                    outview_features = torch.zeros((0, tmp_history_features.shape[1]), dtype=tmp_history_features.dtype, device=tmp_history_features.device)
                    outview_poses = torch.zeros((0, 3), dtype=tmp_history_poses.dtype, device=tmp_history_poses.device)
                else:
                    outview_opacities = voxel_opacities[i][tmp_mask][~new_mask]
                    outview_features = tmp_history_features[~new_mask]
                    outview_poses = tmp_history_poses[~new_mask]
                    voxel_stream_quantile = min(self.cfg['voxel_stream_quantile_outview'], self.cfg['voxel_stream_outview_num']/outview_opacities.shape[0])
                    opacities_thre = torch.quantile(outview_opacities, 1 - voxel_stream_quantile)
                    opacities_thre = max(opacities_thre, self.cfg.voxel_stream_opacity_thre)
                    opacities_mask = outview_opacities > opacities_thre
                    if opacities_mask.sum() == 0:
                        # at least keep one
                        opacities_mask[0] = True
                    outview_features = outview_features[opacities_mask]
                    outview_poses = outview_poses[opacities_mask]

                tmp_history_features = torch.cat([inview_features, outview_features], dim=0)
                tmp_history_poses = torch.cat([inview_poses, outview_poses], dim=0)
                tmp_history_coords = tmp_history_poses - self.pts_range[None, :3]
                tmp_history_coords = torch.floor(tmp_history_coords / (self.voxel_size[None, :3] * 2)).int()
                bidx = tmp_history_coords.new_zeros((tmp_history_coords.shape[0], 1)) + i
                tmp_history_coords = torch.cat([bidx.int(), tmp_history_coords], dim=-1)
                tmp_mask = torch.ones(tmp_history_coords.shape[0], dtype=torch.bool)
            process_voxel_features.append(tmp_history_features)
            process_voxel_coords.append(tmp_history_coords)
            process_voxel_pos.append(tmp_history_poses)
            process_masks.append(tmp_mask)
        
        process_voxel_features = torch.cat(process_voxel_features, dim=0)
        process_voxel_coords = torch.cat(process_voxel_coords, dim=0)
        process_voxel_coords = process_voxel_coords[:, [0, 3, 2, 1]] # bzyx
        process_voxel_pos = torch.cat(process_voxel_pos, dim=0)
        process_masks = torch.cat(process_masks, dim=0).to(images.device)
        history_infos = (process_voxel_features, process_voxel_coords, process_voxel_pos, process_masks)
        gaussians_voxel, point_coords, save_features, save_coords = self.unet(gaussians_voxel, voxel_coords, B, history_infos)
        gaussians_voxel, voxel_batch_idx, voxel_opacities = self.process_guassian_voxel(gaussians_voxel, point_coords[:,1:], voxel_batch_idx=point_coords[:, 0])
        
        # final render parts
        sky_mask = depth_conf_unscale.reshape(B, S, H, W)==0
        img_guassian_features = img_guassian_features.reshape(B, S, -1, H, W)
        save_gaussians, save_gaussians_outview = [], []
        for i in range(B):
            gaussians_source1, gaussians_source1_dynscore, gaussians_sky,  gaussians_sky_dynscore \
                = self.refine_guassians(save_features[save_coords[:,0]==i], save_coords[save_coords[:,0]==i], \
                pixel_means[i], sky_mask[i], img_guassian_features[i])
            # process memory gaussians
            extra_gaussians_outview = self.process_history_gaussians(history_gaussians[i], history_gaussians_outview[i], \
                current_lidar2world[i], history_lidar2world[i], intrinsics[i], camera2lidar[i], gaussians_source1[:,:3].detach(), W, H)
            save_gaussians_outview.append(extra_gaussians_outview.detach())
            # add sky
            gaussians_source1 = torch.cat([gaussians_source1, gaussians_sky], dim=0)
            gaussians_source1_dynscore = torch.cat([gaussians_source1_dynscore, gaussians_sky_dynscore], dim=0)
            batch_mask = voxel_batch_idx == i
            gaussians_source2 = gaussians_voxel[batch_mask][:,:14]
            gaussians_source2_dynscore = gaussians_voxel[batch_mask][:,14:]

            H_stack = [rgb_gt.shape[-2]]* rgb_gt.shape[1]
            W_stack = [rgb_gt.shape[-1]]* rgb_gt.shape[1]
            H_stack = torch.tensor(H_stack, dtype=torch.int, device=device_id)
            W_stack = torch.tensor(W_stack, dtype=torch.int, device=device_id)
            
            gaussians_all = torch.cat([gaussians_source1, gaussians_source2], dim=0)
            gaussians_all_dynscore = torch.cat([gaussians_source1_dynscore, gaussians_source2_dynscore], dim=0)
            tmp = self.renderer.render(
                gaussians=gaussians_all[:, :],
                c2w=render_c2w[i, select_index],
                fovx=render_fovxs[i, select_index],
                fovy=render_fovys[i, select_index],
                rays_o=None,
                rays_d=None,
                K=out_intrinsics[i, select_index],
                H=H_stack[select_index],
                W=W_stack[select_index],
                semantics=gaussians_all_dynscore,
            )
            test_out = tmp
            pred_dynamics_region = (tmp['dyn'][5:] > 0.5)[:,0,:,:]
            dyn_gs_mask = self.get_dynamic_gs_mask(pred_dynamics_region, gaussians_all[:,:3].detach(), intrinsics[i], camera2lidar[i])
            # just save static gaussians
            save_gaussians.append(gaussians_all.detach()[(dyn_gs_mask==False)])
        # cache history
        cache_dict = {'images': images, 'lidars': [None, None],  'scenes': aggregated_tokens_list['scene'], \
                'frames': aggregated_tokens_list['frame'], 'gt_dynamics_mask': [None, None], 'pred_dynamics_mask': [None, None], \
                'cam2lidars': [None, None], 'lidar2worlds': aggregated_tokens_list['lidar2world'], 'voxel_features': save_features.detach(), 
                'voxel_poses': save_coords.detach(), 'voxel_opacities':  voxel_opacities.detach(), \
                'gaussians': save_gaussians, 'gaussians_outview': save_gaussians_outview}
        self.history_queue.cache(B, **cache_dict)

        return test_out, None

    def _forward_impl(self, aggregated_tokens_list, images, patch_start_idx, input_dict_gs, output_dict_gs, stage=3):

        B, S, _, H, W = images.shape

        input_dict_gs['scene'] = aggregated_tokens_list['scene']
        input_dict_gs['frame'] = aggregated_tokens_list['frame']

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        device_id = images.device
        dtype = images.dtype

        local_points = aggregated_tokens_list['local_points']
        local_depth = local_points[..., 2]
        intrinsics = input_dict_gs["intrinsics"].clone()
        sky_far = self.cfg.get('sky_mask', 300.0)

        # pred scale and shift from pi3 intermediate features
        pos = self.position_getter(B * S, patch_h, patch_w, images.device)
        out = []
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list['intermidiate_output'][layer_idx][:, patch_start_idx:]
            out.append(x)
        out = torch.stack(out, dim=-1).mean(dim=-1)
        out_hidden = self.point_decoder(out, xpos=pos)
        out_hidden = out_hidden.mean(dim=1)
        pred_scale = self.scale_head(out_hidden).squeeze(-1).exp().reshape(B, S)
        pred_shift = self.shift_head(out_hidden).squeeze(-1).reshape(B, S)

        # stage 1 only needs scale/shift supervision; skip the rest of the pipeline
        if stage == 1:
            intrinsics_norm = intrinsics.clone()
            for i in range(B):
                for j in range(S):
                    intrinsics_norm[i][j] = normalize_intrinsics(intrinsics_norm[i][j], W, H)
            single_depthmaps = input_dict_gs['single_depthmaps'].to(device_id, dtype)
            gt_mask = (single_depthmaps > 0.1) & (single_depthmaps < sky_far)
            single_depthmaps = torch.where(gt_mask, single_depthmaps, torch.zeros_like(single_depthmaps))
            gt_points = utils3d.torch.depth_to_points(single_depthmaps, intrinsics=intrinsics_norm)
            local_points_3d = utils3d.torch.depth_to_points(local_depth, intrinsics=intrinsics_norm)
            (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize(
                (local_points_3d, gt_points), mask=gt_mask, size=(32, 32))
            gt_scale, gt_shift = align_points_scale_z_shift(
                pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2),
                lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=1.0)
            valid = gt_scale > 0
            gt_scale = torch.where(valid, gt_scale, torch.zeros_like(gt_scale))
            gt_shift_z = torch.where(valid, gt_shift[..., 2], torch.zeros_like(gt_shift[..., 2]))
            valid_f = valid.to(dtype)
            weight = valid_f.sum().clamp_min(1.0)
            scale_loss = ((pred_scale - gt_scale.detach()).abs() * valid_f).sum() / weight
            shift_loss = ((pred_shift - gt_shift_z.detach()).abs() * valid_f).sum() / weight
            return None, {
                'scale_loss': scale_loss * self.cfg.get('scale_loss_weight', 0.1),
                'shift_loss': shift_loss * self.cfg.get('shift_loss_weight', 1.0),
            }

        # stage 2 / 3 share the rendering path
        images_norm = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)
        images_norm = images_norm.view(B * S, 3, H, W)
        dinov2_features = self.image_backbone(images_norm)
        dinov2_features = dinov2_features['x_norm_patchtokens']
        dinov2_features = self.dinov2_proj(dinov2_features)

        rays_o = input_dict_gs["rays_o"].to(device_id, dtype)
        rays_d = input_dict_gs["rays_d"].to(device_id, dtype)
        pluckers = self.plucker_embedder(rays_o, rays_d)
        pluckers = pluckers.reshape(B*S, -1, H, W)
        plucker_embeds = self.plucker_to_embed(pluckers)

        loss_dict = {}
        if stage == 2:
            # supervise depth_map_unscale with gt scale/shift from depth alignment
            intrinsics_norm = intrinsics.clone()
            for i in range(B):
                for j in range(S):
                    intrinsics_norm[i][j] = normalize_intrinsics(intrinsics_norm[i][j], W, H)
            single_depthmaps = input_dict_gs['single_depthmaps'].to(device_id, dtype)
            gt_mask = (single_depthmaps > 0.1) & (single_depthmaps < sky_far)
            single_depthmaps = torch.where(gt_mask, single_depthmaps, torch.zeros_like(single_depthmaps))
            gt_points = utils3d.torch.depth_to_points(single_depthmaps, intrinsics=intrinsics_norm)
            local_points_3d = utils3d.torch.depth_to_points(local_depth, intrinsics=intrinsics_norm)
            (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize(
                (local_points_3d, gt_points), mask=gt_mask, size=(32, 32))
            gt_scale, gt_shift = align_points_scale_z_shift(
                pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2),
                lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=1.0)
            valid = gt_scale > 0
            depth_scale = torch.where(valid, gt_scale, torch.zeros_like(gt_scale)).detach()
            depth_shift = torch.where(valid, gt_shift[..., 2], torch.zeros_like(gt_shift[..., 2])).detach()
        else:
            depth_scale = pred_scale
            depth_shift = pred_shift
        depth_map_unscale = local_depth * depth_scale[:, :, None, None] + depth_shift[:, :, None, None]

        depth_conf_unscale = input_dict_gs['sky_mask']
        # give sky depth
        if self.cfg.get('sky_mask', 0.0) > 0.0:
            depth_map_unscale = torch.where(depth_conf_unscale == 0, \
                depth_map_unscale.new_full(depth_map_unscale.shape, self.cfg['sky_mask']), depth_map_unscale)
        depth_map_unscale = torch.clamp(depth_map_unscale, max=self.cfg['sky_mask'], min=0.0)
        depthmap = torch.cat([depth_map_unscale[:, :, None, :, :]/self.cfg.get('sky_mask', 300.0),\
            depth_conf_unscale[:, :, None, :, :], images], dim=2)
        depthmap = depthmap.reshape(B * S, -1, H, W)
        depth_embeds = self.depth_embeds(depthmap)
        agg_embeds = plucker_embeds + depth_embeds
        agg_embeds = self.embed_proj(agg_embeds)
        agg_embeds = agg_embeds.reshape(B*S, patch_h * patch_w, -1)

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list['intermidiate_output'][layer_idx][:, patch_start_idx:]
            x = x + agg_embeds + dinov2_features
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
        fpn_out_project = self.voxel_proj(out[0])
        out = self.scratch_forward(out)
        out = custom_interpolate(out,
            (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
            mode="bilinear", align_corners=True)
        img_guassian_features = self.feature_norm(out)
        means = rays_o + rays_d * depth_map_unscale[..., None]
        pixel_means = means.view(B, -1, 3)

        # rendering targets
        out_view_num = output_dict_gs["rgb"].shape[1]
        select_mask = torch.zeros((out_view_num), dtype=torch.bool, device=device_id)
        select_mask[:] = True
        select_index = torch.nonzero(select_mask, as_tuple=False).squeeze(-1)
        render_c2w = output_dict_gs["c2w"].to(device_id, dtype)
        render_fovxs = output_dict_gs["fovx"].to(device_id, dtype)
        render_fovys = output_dict_gs["fovy"].to(device_id, dtype)
        rgb_gt = output_dict_gs["rgb"].to(device_id, dtype)
        out_intrinsics = output_dict_gs["intrinsics"].to(device_id, dtype)
        dynamics_region = output_dict_gs["dynamics_region"].to(device_id, dtype)
        input_dynamics_region = input_dict_gs['dynamics_region'].to(device_id, dtype)

        # project current-frame dynamic points to novel views, used to mask dyn regions in rgb loss
        future_num = out_view_num - S
        future_project_masks = images.new_zeros((B, future_num, H, W), dtype=torch.bool)
        for i in range(B):
            dynamic_points = means[i][input_dynamics_region[i] == 1]
            if dynamic_points.shape[0] == 0 or future_num == 0:
                continue
            future_c2w = render_c2w[i, :future_num]
            future_intrinsics = out_intrinsics[i, :future_num]
            mask, pos2d = is_point_in_frustum_batch(dynamic_points, future_intrinsics, \
                future_c2w[:, :3, :3], future_c2w[:, :3, 3], 0.5, sky_far, W, H)
            for j in range(future_num):
                valid_pos2d = pos2d[:, j][mask[:, j]]
                future_project_masks[i, j, valid_pos2d[:, 1].long(), valid_pos2d[:, 0].long()] = True

        # init voxel
        self.pts_range = self.pts_range.to(device=means.device)
        self.voxel_size = self.voxel_size.to(device=means.device)
        coords = []
        multimodel_gaussians = []
        for i in range(B):
            means = pixel_means[i]
            boundary = self.pts_range.clone()
            boundary[:3] = boundary[:3] + 0.01
            boundary[3:6] = boundary[3:6] - 0.01
            mask = (means[:, 0] > boundary[0]) & (means[:, 0] < boundary[3]) & \
                   (means[:, 1] > boundary[1]) & (means[:, 1] < boundary[4]) & \
                   (means[:, 2] > boundary[2]) & (means[:, 2] < boundary[5])
            count = mask.sum()
            batch_idx = means.new_zeros((count, 1)) + i
            batch_idx = batch_idx.int()
            gaussians_coords = means[mask]
            gaussians_coords = gaussians_coords - self.pts_range[None, :3]
            gaussians_coords = torch.floor(gaussians_coords / self.voxel_size[None, :3]).int()
            coords.append(torch.cat([batch_idx, gaussians_coords], dim=-1))
            rgbs = images[i].permute(0,2,3,1).reshape(-1, 3)
            multimodel_gaussians.append(torch.cat([means[mask], rgbs[mask]], dim=-1))

        coords = torch.cat(coords, dim=0)
        multimodel_gaussians = torch.cat(multimodel_gaussians, dim=0)
        merge_coords = coords[:, 0].int() * self.scale_xyz + coords[:, 1] * self.scale_yz + \
                       coords[:, 2] * self.scale_z + coords[:, 3]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)
        gaussians_voxel = torch_scatter.scatter_mean(multimodel_gaussians, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]] # bzyx
        voxel_centers = get_voxel_centers(voxel_coords[:,1:], 1, self.voxel_size, self.pts_range)
        # extract image features
        intrinsics = input_dict_gs["intrinsics"]
        camera2lidar = input_dict_gs["camera2lidar"]
        gaussians_imgfeats = []
        for i in range(B):
            batch_mask = voxel_coords[:, 0] == i
            points = voxel_centers[batch_mask]
            projected_points, z_cam, valid_masks = project_world_points_to_images(points, intrinsics[i], camera2lidar[i], H, W)
            image_modality = fpn_out_project
            projected_points[:, :, 0] = (projected_points[:, :, 0] / W) * 2 - 1
            projected_points[:, :, 1] = (projected_points[:, :, 1] / H) * 2 - 1
            out_features = image_modality.new_zeros((image_modality.shape[0], projected_points.shape[1], image_modality.shape[1]))
            for view in range(valid_masks.shape[0]):
                input_features = image_modality[i*S+view:i*S+view+1]
                input_points = projected_points[view:view+1]
                input_mask = valid_masks[view:view+1]
                valid_points = input_points[input_mask]
                tmp_features = F.grid_sample(
                    input_features,
                    valid_points[None, None, :, :],
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )
                out_features[view][input_mask[0]] = tmp_features[0,:,0,:].transpose(0, 1)
            out_features = out_features.sum(dim=0) / (valid_masks.sum(dim=0)[:,None] + 1e-6)
            gaussians_imgfeats.append(out_features)
        gaussians_imgfeats = torch.cat(gaussians_imgfeats, dim=0)
        gaussians_voxel = torch.cat([gaussians_voxel, gaussians_imgfeats], dim=-1)

        # get history
        _, _, _, _, _, history_lidar2world, history_features, history_poses, voxel_opacities, \
            history_gaussians, history_gaussians_outview \
            = self.history_queue.get(B, aggregated_tokens_list['scene'], aggregated_tokens_list['frame'])
        current_lidar2world = aggregated_tokens_list['lidar2world']
        process_voxel_features, process_voxel_coords, process_voxel_pos, process_masks = [], [], [], []
        for i in range(B):
            if history_poses[i] is None:
                tmp_history_features = torch.zeros((1, 64), dtype=gaussians_voxel.dtype, device=gaussians_voxel.device)
                tmp_history_poses = voxel_centers[voxel_coords[:,0] == i][:1]
                tmp_history_coords = tmp_history_poses - self.pts_range[None, :3]
                tmp_history_coords = torch.floor(tmp_history_coords / (self.voxel_size[None, :3] * 2)).int()
                bidx = tmp_history_coords.new_zeros((tmp_history_coords.shape[0], 1)) + i
                tmp_history_coords = torch.cat([bidx.int(), tmp_history_coords], dim=-1)
                tmp_mask = torch.tensor([False])
            else:
                history_2_current = (current_lidar2world[i].inverse() @ history_lidar2world[i])[0]
                ones = torch.ones((history_poses[i].shape[0], 1), dtype=images.dtype, device=images.device)
                history_poses_homo = torch.concat([history_poses[i], ones], dim=-1)
                history_poses_homo = (history_2_current @ history_poses_homo.T).T
                history_poses_homo = history_poses_homo[:,:-1]
                tmp_mask = (history_poses_homo[:, 0] > boundary[0]) & (history_poses_homo[:, 0] < boundary[3]) & \
                   (history_poses_homo[:, 1] > boundary[1]) & (history_poses_homo[:, 1] < boundary[4]) & \
                   (history_poses_homo[:, 2] > boundary[2]) & (history_poses_homo[:, 2] < boundary[5])

                tmp_history_poses = history_poses_homo[tmp_mask]
                tmp_history_features = history_features[i][tmp_mask]
                new_mask, _ = is_point_in_frustum_batch(tmp_history_poses, intrinsics[i], camera2lidar[i,:, :3,:3], camera2lidar[i,:, :3,3], 0.5, 72, 518, 350)
                new_mask = new_mask.any(dim=-1)
                inview_opacities = voxel_opacities[i][tmp_mask][new_mask]
                inview_features = tmp_history_features[new_mask]
                inview_poses = tmp_history_poses[new_mask]
                voxel_stream_quantile = min(self.cfg['voxel_stream_quantile'], self.cfg['voxel_stream_inview_num']/inview_opacities.shape[0])
                opacities_thre = torch.quantile(inview_opacities, 1 - voxel_stream_quantile)
                opacities_thre = max(opacities_thre, self.cfg.voxel_stream_opacity_thre)
                opacities_mask = inview_opacities > opacities_thre
                if opacities_mask.sum() == 0:
                    opacities_mask[0] = True
                inview_features = inview_features[opacities_mask]
                inview_poses = inview_poses[opacities_mask]
                if (~new_mask).sum() == 0:
                    outview_features = torch.zeros((0, tmp_history_features.shape[1]), dtype=tmp_history_features.dtype, device=tmp_history_features.device)
                    outview_poses = torch.zeros((0, 3), dtype=tmp_history_poses.dtype, device=tmp_history_poses.device)
                else:
                    outview_opacities = voxel_opacities[i][tmp_mask][~new_mask]
                    outview_features = tmp_history_features[~new_mask]
                    outview_poses = tmp_history_poses[~new_mask]
                    voxel_stream_quantile = min(self.cfg['voxel_stream_quantile_outview'], self.cfg['voxel_stream_outview_num']/outview_opacities.shape[0])
                    opacities_thre = torch.quantile(outview_opacities, 1 - voxel_stream_quantile)
                    opacities_thre = max(opacities_thre, self.cfg.voxel_stream_opacity_thre)
                    opacities_mask = outview_opacities > opacities_thre
                    if opacities_mask.sum() == 0:
                        opacities_mask[0] = True
                    outview_features = outview_features[opacities_mask]
                    outview_poses = outview_poses[opacities_mask]

                tmp_history_features = torch.cat([inview_features, outview_features], dim=0)
                tmp_history_poses = torch.cat([inview_poses, outview_poses], dim=0)
                tmp_history_coords = tmp_history_poses - self.pts_range[None, :3]
                tmp_history_coords = torch.floor(tmp_history_coords / (self.voxel_size[None, :3] * 2)).int()
                bidx = tmp_history_coords.new_zeros((tmp_history_coords.shape[0], 1)) + i
                tmp_history_coords = torch.cat([bidx.int(), tmp_history_coords], dim=-1)
                tmp_mask = torch.ones(tmp_history_coords.shape[0], dtype=torch.bool)
            process_voxel_features.append(tmp_history_features)
            process_voxel_coords.append(tmp_history_coords)
            process_voxel_pos.append(tmp_history_poses)
            process_masks.append(tmp_mask)

        process_voxel_features = torch.cat(process_voxel_features, dim=0)
        process_voxel_coords = torch.cat(process_voxel_coords, dim=0)
        process_voxel_coords = process_voxel_coords[:, [0, 3, 2, 1]] # bzyx
        process_voxel_pos = torch.cat(process_voxel_pos, dim=0)
        process_masks = torch.cat(process_masks, dim=0).to(images.device)
        history_infos = (process_voxel_features, process_voxel_coords, process_voxel_pos, process_masks)
        gaussians_voxel, point_coords, save_features, save_coords = self.unet(gaussians_voxel, voxel_coords, B, history_infos)
        gaussians_voxel, voxel_batch_idx, voxel_opacities = self.process_guassian_voxel(gaussians_voxel, point_coords[:,1:], voxel_batch_idx=point_coords[:, 0])

        sky_mask = depth_conf_unscale.reshape(B, S, H, W) == 0
        img_guassian_features = img_guassian_features.reshape(B, S, -1, H, W)
        loss_dict_all = {}
        save_gaussians, save_gaussians_outview = [], []
        for i in range(B):
            gaussians_source1, gaussians_source1_dynscore, gaussians_sky,  gaussians_sky_dynscore \
                = self.refine_guassians(save_features[save_coords[:,0]==i], save_coords[save_coords[:,0]==i], \
                pixel_means[i], sky_mask[i], img_guassian_features[i])
            extra_gaussians_outview = self.process_history_gaussians(history_gaussians[i], history_gaussians_outview[i], \
                current_lidar2world[i], history_lidar2world[i], intrinsics[i], camera2lidar[i], gaussians_source1[:,:3].detach(), W, H)
            save_gaussians_outview.append(extra_gaussians_outview.detach())
            # add sky
            gaussians_source1 = torch.cat([gaussians_source1, gaussians_sky], dim=0)
            gaussians_source1_dynscore = torch.cat([gaussians_source1_dynscore, gaussians_sky_dynscore], dim=0)
            batch_mask = voxel_batch_idx == i
            gaussians_source2 = gaussians_voxel[batch_mask][:,:14]
            gaussians_source2_dynscore = gaussians_voxel[batch_mask][:,14:]

            H_stack = [rgb_gt.shape[-2]]* rgb_gt.shape[1]
            W_stack = [rgb_gt.shape[-1]]* rgb_gt.shape[1]
            H_stack = torch.tensor(H_stack, dtype=torch.int, device=device_id)
            W_stack = torch.tensor(W_stack, dtype=torch.int, device=device_id)

            gaussians_all = torch.cat([gaussians_source1, gaussians_source2], dim=0)
            gaussians_all_dynscore = torch.cat([gaussians_source1_dynscore, gaussians_source2_dynscore], dim=0)
            tmp = self.renderer.render(
                gaussians=gaussians_all[:, :],
                c2w=render_c2w[i, select_index],
                fovx=render_fovxs[i, select_index],
                fovy=render_fovys[i, select_index],
                rays_o=None,
                rays_d=None,
                K=out_intrinsics[i, select_index],
                H=H_stack[select_index],
                W=W_stack[select_index],
                semantics=gaussians_all_dynscore,
            )
            test_out = tmp
            dyn_mask = (dynamics_region[i:i+1].clone() == 1)
            if future_num > 0:
                dyn_mask[:, :future_num] = dyn_mask[:, :future_num] | future_project_masks[i:i+1]
            tmp_loss = self.compute_loss(tmp["image"], tmp['dyn'], rgb_gt[i], select_index, \
                dyn_mask=dyn_mask[0], dyn_gt=input_dynamics_region[i])
            for k, v in tmp_loss.items():
                if 'all_' + k not in loss_dict_all:
                    loss_dict_all['all_'+k] = v
                else:
                    loss_dict_all['all_'+k] += v

            # voxel-branch only auxiliary rgb loss (reconstruction views, restricted to in-range pixels)
            if self.cfg.get('aux_voxel_weight', 0.0) > 0.0:
                pixel_mask = pixel_means[i]
                pixel_mask = (pixel_mask > self.pts_range[None, :3]) & (pixel_mask < self.pts_range[None, 3:])
                pixel_mask = pixel_mask.view(S, H, W, 3).all(dim=-1)
                aux_tmp = self.renderer.render(
                    gaussians=gaussians_source2[:, :], c2w=render_c2w[i, future_num:],
                    fovx=render_fovxs[i, future_num:], fovy=render_fovys[i, future_num:],
                    rays_o=None, rays_d=None,
                    K=out_intrinsics[i, future_num:], H=H_stack[future_num:], W=W_stack[future_num:],
                    semantics=gaussians_source2_dynscore,
                )
                aux_voxel_loss = (aux_tmp["image"] - rgb_gt[i][future_num:]) ** 2
                aux_voxel_loss = aux_voxel_loss * pixel_mask[:, None, :, :]
                aux_voxel_loss = aux_voxel_loss.mean() * self.cfg.get('aux_voxel_weight', 0.0)
                if 'all_aux_voxel_loss' not in loss_dict_all:
                    loss_dict_all['all_aux_voxel_loss'] = aux_voxel_loss
                else:
                    loss_dict_all['all_aux_voxel_loss'] += aux_voxel_loss

            pred_dynamics_region = (tmp['dyn'][future_num:] > self.cfg.get('dyn_save_thre', 0.2))[:,0,:,:].detach()
            dyn_gs_mask = self.get_dynamic_gs_mask(pred_dynamics_region, gaussians_all[:,:3].detach(), intrinsics[i], camera2lidar[i])
            save_gaussians.append(gaussians_all.detach()[(dyn_gs_mask==False)])

        for k, v in loss_dict_all.items():
            loss_dict_all[k] = v / B
        loss_dict.update(loss_dict_all)

        cache_dict = {'images': images, 'lidars': [None, None],  'scenes': aggregated_tokens_list['scene'], \
                'frames': aggregated_tokens_list['frame'], 'gt_dynamics_mask': [None, None], 'pred_dynamics_mask': [None, None], \
                'cam2lidars': [None, None], 'lidar2worlds': aggregated_tokens_list['lidar2world'], 'voxel_features': save_features.detach(),
                'voxel_poses': save_coords.detach(), 'voxel_opacities':  voxel_opacities.detach(), \
                'gaussians': save_gaussians, 'gaussians_outview': save_gaussians_outview}
        self.history_queue.cache(B, **cache_dict)

        return test_out, loss_dict

    def compute_loss(self, rgb_render, dyn_render, rgb_gt, select_index, dyn_mask=None, dyn_gt=None):
        loss_dict = {}
        S, C, H, W = rgb_gt.shape
        rgb_gt = rgb_gt[select_index]
        rec_loss = (rgb_gt - rgb_render) ** 2
        if dyn_mask is not None:
            static_mask = (dyn_mask == 0).to(rec_loss.dtype)
            rec_loss = rec_loss * static_mask[select_index, None, :, :]
        dtype = rgb_gt.dtype
        device_id = rgb_gt.device
        if self.l1_loss_mask is not None:
            mask = torch.tensor(self.l1_loss_mask, dtype=dtype, device=device_id)
            mask = mask[select_index]
            rec_loss = rec_loss * mask[:, None, None, None]
        rec_loss = rec_loss.mean()
        loss_dict["rec_loss"] = rec_loss * 1.0 if self.loss_weight is None else rec_loss * self.loss_weight['rec_loss']

        # perceptual loss
        p_loss = self.perceptual_loss(rgb_render.reshape(-1, 3, H, W), rgb_gt.reshape(-1, 3, H, W))
        if self.p_loss_mask is not None:
            mask = torch.tensor(self.p_loss_mask, dtype=dtype, device=device_id)
            mask = mask[select_index]
            p_loss = p_loss * mask[:, None, None, None]
        p_loss = p_loss.mean()
        loss_dict["perceptual_loss"] = p_loss * 0.05 if self.loss_weight is None else p_loss * self.loss_weight['perceptual_loss']

        # dynamic head loss (BCE on input-view rendered dyn vs gt dyn region)
        S_in = dyn_gt.shape[0]
        dyn_render = dyn_render[-S_in:, 0]
        dyn_gt = dyn_gt.flatten()
        dyn_render = dyn_render.flatten()
        if dyn_gt.sum() == 0:
            sampled_neg_indices = torch.randperm(dyn_gt.shape[0], device=dyn_gt.device)
            sampled_neg_indices = sampled_neg_indices[:self.cfg.neg_dyn_num]
            dyn_loss = F.binary_cross_entropy(dyn_render[sampled_neg_indices], dyn_gt[sampled_neg_indices], reduction='mean')
        else:
            pos_indices = torch.where(dyn_gt == 1)[0]
            neg_indices = torch.where(dyn_gt == 0)[0]
            n_neg_samples = len(pos_indices) * self.cfg.neg_dyn_ratio
            sampled_neg_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg_samples]]
            train_indices = torch.cat([pos_indices, sampled_neg_indices])
            dyn_loss = F.binary_cross_entropy(dyn_render[train_indices], dyn_gt[train_indices], reduction='mean')
        loss_dict["dyn_loss"] = self.cfg.dyn_loss_weight * dyn_loss

        return loss_dict

    def process_guassian_voxel(self, gaussians, means, voxel_batch_idx):
        gaussians = gaussians.reshape(-1, self.cfg.voxel_gs_num, 15)
        means = means[:, None, :].repeat(1, self.cfg.voxel_gs_num, 1)
        gaussians = gaussians.reshape(-1, 15)
        means = means.reshape(-1, 3)
        voxel_batch_idx = voxel_batch_idx[:, None].repeat(1, self.cfg.voxel_gs_num)
        voxel_batch_idx = voxel_batch_idx.reshape(-1)
        
        offsets = gaussians[..., :3]
        opacities = self.opt_act(gaussians[..., 3:4])
        scales = self.scale_act(gaussians[..., 4:7])
        scales = torch.clamp_max(scales, self.cfg['max_scale_voxel'])
        rotations = self.rot_act(gaussians[..., 7:11])
        rgbs = self.rgb_act(gaussians[..., 11:14])
        dynamic = nn.functional.sigmoid(gaussians[..., 14:15])
        means = means + offsets
        gaussians = torch.cat([means, rgbs, opacities, rotations, scales, dynamic], dim=-1)
        voxel_opacities = opacities[:,0].reshape(-1, self.cfg.voxel_gs_num)
        voxel_opacities = voxel_opacities.mean(dim=-1)
        return gaussians, voxel_batch_idx, voxel_opacities
    
    def plucker_embedder(self, rays_o, rays_d):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out
    
    def refine_guassians(self, voxel_features, voxel_coords, means, sky_mask, img_features):
        from simple_knn_v2._C import distCUDACross
        
        means_in = means - self.pts_range[None, :3]
        means_in = torch.floor(means_in / (self.voxel_size[None, :3] * 2)).int()
        means_in = (means_in + 0.5) * self.voxel_size[None, :3] * 2 + self.pts_range[None, :3]
        points = torch.cat([means_in, voxel_coords[:, 1:]], dim=0)
        labels = torch.zeros((points.shape[0],), dtype=torch.int, device=means_in.device)
        labels[means_in.shape[0]:] = 1
        dist, indices = distCUDACross(points, labels)
        voxel_indices = indices[:means_in.shape[0]] - means_in.shape[0]
        has_voxel_mask = dist[:means_in.shape[0]]==0
        sample_features = voxel_features.new_zeros(means.shape[0], voxel_features.shape[1])
        sample_features[has_voxel_mask] = voxel_features[voxel_indices[has_voxel_mask]]
        img_features = img_features.permute(0,2,3,1).reshape(-1, 256)
        
        sample_features = torch.cat([sample_features, img_features], dim=-1)
        
        sky_mask = sky_mask.flatten()
        gaussians = self.to_gaussians(sample_features[~sky_mask])
        gaussians_sky = self.to_gaussians_sky(sample_features[sky_mask])
        
        offsets = self.offset_act(gaussians[..., :3])
        opacities = self.opt_act(gaussians[..., 3:4])
        scales = self.scale_act(gaussians[..., 4:7])
        scales = torch.clamp_max(scales, self.cfg.get('max_scale', 0.01))
        rotations = self.rot_act(gaussians[..., 7:11])
        rgbs = self.rgb_act(gaussians[..., 11:14])
        dynamic = nn.functional.sigmoid(gaussians[..., 14:15])
        common_means = means[~sky_mask]
        common_means = common_means + offsets
        gaussians = torch.cat([common_means, rgbs, opacities, rotations, scales], dim=-1)
        
        sky_offsets = nn.functional.tanh(gaussians_sky[..., :3]) * self.cfg.get('sky_gaussian_offset', 20)
        sky_opacities = self.opt_act(gaussians_sky[..., 3:4])
        sky_scales = self.scale_act(gaussians_sky[..., 4:7])
        sky_scales = torch.clamp_max(sky_scales, self.cfg.sky_gaussian_scale)
        sky_rotations = self.rot_act(gaussians_sky[..., 7:11])
        sky_rgbs = self.rgb_act(gaussians_sky[..., 11:14])
        sky_dynamic = nn.functional.sigmoid(gaussians_sky[..., 14:15])
        sky_means = means[sky_mask]
        sky_means = sky_means + sky_offsets
        sky_gaussians = torch.cat([sky_means, sky_rgbs, sky_opacities, sky_rotations, sky_scales], dim=-1)
        
        return gaussians, dynamic, sky_gaussians, sky_dynamic

    def get_dynamic_gs_mask(self, dynamics_region, means, intrinsics, camera2lidars):
        S, H, W = dynamics_region.shape
        dynamics_region_num = dynamics_region.flatten(start_dim=1).sum(dim=-1)
        out_mask = means.new_zeros((means.shape[0]), dtype=torch.bool)
        for i in range(S):
            if dynamics_region_num[i] == 0:
                continue
            inview_mask, pos2d = is_point_in_frustum_batch(means, intrinsics[i:i+1], camera2lidars[i:i+1, :3, :3], camera2lidars[i:i+1, :3, 3], 0.1, 300, W, H)
            inview_mask = inview_mask[:, 0]
            pos2d = pos2d[:, 0, :]
            pos2d = pos2d[inview_mask]
            d_mask = dynamics_region[i][pos2d[:, 1], pos2d[:, 0]]
            out_mask[inview_mask] = out_mask[inview_mask] | d_mask.to(torch.bool)
        return out_mask

    def process_history_gaussians(self, history_gaussians, history_gaussians_outview, current_lidar2world, history_lidar2world, \
                                  intrinsics, camera2lidars, current_means, W, H):
        dtype, device = current_means.dtype, current_means.device
        if history_gaussians is None:
            history_gaussians_outview_save = torch.zeros((0, 14), dtype=dtype, device=device)
        else:
            history_2_current = (current_lidar2world.inverse() @ history_lidar2world)[0]
            means = history_gaussians[:, :3]
            means_outview = history_gaussians_outview[:, :3]
            means = torch.cat([means, torch.ones((means.shape[0], 1), dtype=dtype, device=device)], dim=-1)
            means_outview = torch.cat([means_outview, torch.ones((means_outview.shape[0], 1), dtype=dtype, device=device)], dim=-1)
            means = (history_2_current @ means.T).T
            means = means[:,:-1]
            means_outview = (history_2_current @ means_outview.T).T
            means_outview = means_outview[:,:-1]
            
            inview_mask, _ = is_point_in_frustum_batch(means, intrinsics, camera2lidars[:, :3, :3], camera2lidars[ :, :3, 3], 0.1, 300, W, H)
            inview_mask = inview_mask.any(dim=-1)
            
            history_gaussians[:,:3] = means
            history_gaussians_outview[:, :3] = means_outview
            history_gaussians_outview_save = torch.cat([history_gaussians[~inview_mask], history_gaussians_outview], dim=0)

        return history_gaussians_outview_save


class Gaussians_Queue_v2(nn.Module):
    def __init__(self):
        super(Gaussians_Queue_v2, self).__init__()
        self.cache_fields = [
            'images', 'gt_dynamics_mask', 'pred_dynamics_mask', 'lidars', 
            'cam2lidars', 'lidar2worlds', 'voxel_features', 'voxel_poses', 
            'voxel_opacities', 'gaussians', 'gaussians_outview'
        ]
        self.voxel_fields = ['voxel_features', 'voxel_poses', 'voxel_opacities']

        for field in self.cache_fields + ['scenes', 'frames']:
            setattr(self, field, None)
    
    def get(self, batch_size, scenes, frames):
        result = {field: [None] * batch_size for field in self.cache_fields}
        
        # first iter
        if self.scenes is None:
            return tuple(result[field] for field in self.cache_fields)
        
        for i in range(batch_size):
            if (self.scenes[i] == scenes[i] and 
                int(self.frames[i]) == int(frames[i]) - 1):
                for field in self.cache_fields:
                    cache_data = getattr(self, field)
                    if cache_data is not None:
                        result[field][i] = cache_data[i]
        
        return tuple(result[field] for field in self.cache_fields)
    
    def cache(self, batch_size, **kwargs):
        scenes = kwargs.pop('scenes')
        frames = kwargs.pop('frames')
        
        if self.scenes is None:
            for field in self.cache_fields + ['scenes', 'frames']:
                setattr(self, field, [None] * batch_size)
        
        for i in range(batch_size):
            self.scenes[i] = scenes[i]
            self.frames[i] = frames[i]
            
            for field in self.cache_fields:
                if field in self.voxel_fields:
                    continue 
                
                data = kwargs.get(field)
                if data is not None:
                    getattr(self, field)[i] = data[i].detach() if isinstance(data[i], torch.Tensor) else data[i]
        
        voxel_poses = kwargs.get('voxel_poses')
        if voxel_poses is not None:
            for i in range(batch_size):
                mask = voxel_poses[:, 0] == i
                for field in self.voxel_fields:
                    data = kwargs.get(field)
                    if data is not None:
                        if field == 'voxel_poses':
                            getattr(self, field)[i] = data[mask][:, 1:]
                        else:
                            getattr(self, field)[i] = data[mask]
