"""UniSplat's original layers with a stateless, current-six-view execution path."""
import torch
import torch.nn.functional as F
import torch_scatter
import utils3d

from .head import GuassianHead
from .utils import align_points_scale_z_shift, mask_aware_nearest_resize
from ..layers.spconv_unet import get_voxel_centers, project_world_points_to_images
from omniscene.geometry import camera_rays, matching_voxel_features


class StaticGaussianHead(GuassianHead):
    def __init__(self, cfg):
        super().__init__(dim_in=2048, cfg=cfg)
        # No queue is present in this model, even an initially empty one.
        del self.history_queue
        self.to_gaussians_sky.requires_grad_(False)

    def forward(self, pi3_output, images, context, stage=3, metric_depth=None):
        b, s, _, hp, wp = images.shape
        h, w = context['image'].shape[-2:]
        if b != 1 or s != 6 or stage not in (1, 2, 3):
            raise ValueError('Static OmniScene expects batch=1, six views and stage 1/2/3')
        if stage in (1, 2) and metric_depth is None:
            raise ValueError('Stage 1/2 requires six input Metric3D depth maps')
        if stage == 3 and metric_depth is not None:
            raise ValueError('Stage 3 / inference must not consume depth supervision')
        intermediate = [x.float() for x in pi3_output['intermidiate_output']]
        tokens = torch.stack([intermediate[i][:, 5:] for i in self.intermediate_layer_idx], -1).mean(-1)
        pos = self.position_getter(b * s, hp // 14, wp // 14, images.device)
        hidden = self.point_decoder(tokens, xpos=pos).mean(1)
        pred_scale = self.scale_head(hidden).squeeze(-1).exp().reshape(b, s)
        pred_shift = self.shift_head(hidden).squeeze(-1).reshape(b, s)
        local_depth = pi3_output['local_points'][..., 2].float()
        alignment = None
        if stage in (1, 2):
            alignment = self.align_depth(local_depth[..., :h, :w], metric_depth, context['intrinsics'])
        result = dict(pred_scale=pred_scale, pred_shift=pred_shift, alignment=alignment)
        if stage == 1:
            return result
        if stage == 2:
            scale, shift, valid = alignment
            # An invalid camera alignment does not turn its entire image into Z=0.
            scale = torch.where(valid, scale, pred_scale.detach())
            shift = torch.where(valid, shift, pred_shift.detach())
        else:
            scale, shift = pred_scale, pred_shift
        depth = (local_depth * scale[..., None, None] + shift[..., None, None]).clamp(0, self.cfg.depth_max_m)
        if not torch.isfinite(depth).all():
            raise FloatingPointError('Nonfinite predicted geometry')

        image_norm = (images - self._resnet_mean.to(images)) / self._resnet_std.to(images)
        dino = self.image_backbone(image_norm.flatten(0, 1))['x_norm_patchtokens']
        dino = self.dinov2_proj(dino)
        origins, directions = camera_rays(context['intrinsics'], context['extrinsics'], hp, wp)
        plucker = self.plucker_embedder(origins, directions).reshape(b * s, 6, hp, wp)
        embed = self.plucker_to_embed(plucker)
        depth_rgb = torch.cat((depth[:, :, None] / self.cfg.depth_max_m,
                               torch.ones_like(depth[:, :, None]), images), 2).flatten(0, 1)
        embed = self.embed_proj(embed + self.depth_embeds(depth_rgb))
        pyramid = []
        for i, index in enumerate(self.intermediate_layer_idx):
            x = self.norm(intermediate[index][:, 5:] + embed + dino)
            x = x.permute(0, 2, 1).reshape(b * s, -1, hp // 14, wp // 14)
            x = self.projects[i](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, wp, hp)
            pyramid.append(self.resize_layers[i](x))
        voxel_image_features = self.voxel_proj(pyramid[0])
        dense = self.scratch_forward(pyramid)
        dense = F.interpolate(dense, (hp, wp), mode='bilinear', align_corners=True)
        # Cropping precedes point anchors, voxelization and all geometry losses.
        dense = dense[..., :h, :w].reshape(b, s, 256, h, w)
        points = (origins + directions * depth[..., None])[:, :, :h, :w].contiguous()
        rgb = context['image'].permute(0, 1, 3, 4, 2).reshape(-1, 3)
        means = points.reshape(-1, 3)
        voxel_raw, voxel_centers, features, centers = self.scaffold(
            means, rgb, voxel_image_features, context, (hp, wp))
        self.pts_range = self.pts_range.to(means)
        self.voxel_size = self.voxel_size.to(means)
        sampled = matching_voxel_features(means, features, centers, self.pts_range[:3],
                                          self.voxel_size, self.grid_size.to(means.device))
        point_features = torch.cat((sampled, dense.permute(0, 1, 3, 4, 2).reshape(-1, 256)), -1)
        raw = self.to_gaussians(point_features)
        point_gaussians = torch.cat((means + self.offset_act(raw[:, :3]), self.rgb_act(raw[:, 11:14]),
                                     self.opt_act(raw[:, 3:4]), self.rot_act(raw[:, 7:11]),
                                     self.scale_act(raw[:, 4:7]).clamp_max(self.cfg.max_scale)), -1)
        if len(voxel_raw):
            voxels, _, _ = self.process_guassian_voxel(
                voxel_raw, voxel_centers, voxel_raw.new_zeros(len(voxel_raw)))
            voxel_gaussians = voxels[:, :14]
        else:
            voxel_gaussians = point_gaussians.new_empty((0, 14))
        result.update(gaussians=torch.cat((point_gaussians, voxel_gaussians), 0),
                      voxel_gaussians=voxel_gaussians, anchor_points=points,
                      num_point_gaussians=len(point_gaussians), num_voxel_gaussians=len(voxel_gaussians))
        return result

    @torch.no_grad()
    def align_depth(self, local_depth, metric_depth, intrinsics):
        """Original robust scale/Z-shift fit, restricted to real (unpadded) pixels."""
        b, s, h, w = local_depth.shape
        valid_depth = (torch.isfinite(metric_depth) & (metric_depth > .1)
                       & (metric_depth < self.cfg.depth_max_m) & torch.isfinite(local_depth))
        k = intrinsics.clone()
        k[..., 0, :] /= w
        k[..., 1, :] /= h
        target = utils3d.torch.depth_to_points(torch.where(valid_depth, metric_depth, 0), intrinsics=k)
        source = utils3d.torch.depth_to_points(torch.where(valid_depth, local_depth, 0), intrinsics=k)
        (source, target), mask = mask_aware_nearest_resize((source, target), mask=valid_depth, size=(32, 32))
        source, target = source.reshape(b * s, -1, 3), target.reshape(b * s, -1, 3)
        weights = mask.reshape(b * s, -1) / target[..., 2].clamp_min(1e-2)
        usable = (weights > 0).sum(-1) >= 2
        scale = local_depth.new_zeros(b * s)
        shift = local_depth.new_zeros(b * s)
        if usable.any():
            a, t = align_points_scale_z_shift(source[usable], target[usable], weights[usable], trunc=1.0)
            scale[usable], shift[usable] = a, t[..., 2]
        valid = usable & torch.isfinite(scale) & torch.isfinite(shift) & (scale > 0)
        scale = torch.where(valid, scale, 0)
        shift = torch.where(valid, shift, 0)
        # Lack of a fitted teacher is a per-camera supervision state. Stage 1
        # skips an unsupervised update; stage 2 already falls back to predictions.
        return scale.reshape(b, s), shift.reshape(b, s), valid.reshape(b, s)

    def scaffold(self, means, rgb, image_features, context, padded_shape):
        device = means.device
        self.pts_range = self.pts_range.to(device)
        self.voxel_size = self.voxel_size.to(device)
        lower, upper = self.pts_range[:3], self.pts_range[3:]
        inside = ((means > lower + .01) & (means < upper - .01)).all(-1)
        if not inside.any():
            return (means.new_empty((0, 15 * self.cfg.voxel_gs_num)), means.new_empty((0, 3)),
                    means.new_empty((0, 64)), means.new_empty((0, 3)))
        xyz = torch.floor((means[inside] - lower) / self.voxel_size).long()
        shape = self.grid_size.to(device).long()
        keys = xyz[:, 0] * shape[1] * shape[2] + xyz[:, 1] * shape[2] + xyz[:, 2]
        unique, inv = torch.unique(keys, return_inverse=True)
        multimodal = torch_scatter.scatter_mean(torch.cat((means[inside], rgb[inside]), -1), inv, dim=0)
        xyz = torch.stack((unique // (shape[1] * shape[2]), (unique // shape[2]) % shape[1], unique % shape[2]), -1)
        coordinates = torch.cat((xyz.new_zeros((len(xyz), 1)), xyz[:, [2, 1, 0]]), -1).int()
        centers = get_voxel_centers(coordinates[:, 1:], 1, self.voxel_size, self.pts_range)
        h, w = context['image'].shape[-2:]
        hp, wp = padded_shape
        uv, _, visible = project_world_points_to_images(centers, context['intrinsics'][0],
                                                        context['extrinsics'][0], h, w)
        image_sum = means.new_zeros((len(centers), image_features.shape[1]))
        for view in range(6):
            mask = visible[view]
            if not mask.any():
                continue
            grid = uv[view, mask].clone()
            # Feature map covers the padded canvas; validity still uses the original FOV.
            grid[:, 0] = grid[:, 0] / wp * 2 - 1
            grid[:, 1] = grid[:, 1] / hp * 2 - 1
            samples = F.grid_sample(image_features[view:view + 1], grid[None, None],
                                    mode='bilinear', padding_mode='zeros', align_corners=True)
            image_sum[mask] += samples[0, :, 0].T
        image_sum = image_sum / (visible.sum(0)[:, None] + 1e-6)
        features = torch.cat((multimodal, image_sum), -1)
        raw, positions, saved_features, saved_positions = self.unet(features, coordinates, 1, history_infos=None)
        return raw, positions[:, 1:], saved_features, saved_positions[:, 1:]
