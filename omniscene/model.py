"""Build and freeze the reviewed model without changing the original Waymo entry."""
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file

from .config import REPO_ROOT
from .geometry import pad_images

SCALE_MODULES = ('point_decoder', 'scale_head', 'shift_head')


def local_path(path):
    if path is None:
        raise ValueError('A local weight path is required; run tools/download_omniscene_weights.py')
    p = Path(path).expanduser()
    p = p if p.is_absolute() else REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.resolve()


class StaticUniSplat(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        self.checkpoint_rendering = cfg.Model.checkpoint_rendering
        from pi3.models.pi3 import Pi3
        from model.gaussian_head.static_head import StaticGaussianHead
        self.geometry_model = Pi3(decoder_size=cfg.Model.pi3_decoder_size)
        if initialize:
            self.geometry_model.load_state_dict(load_file(str(local_path(cfg.Model.pi3_ckpt))), strict=True)
        self.gaussian_head = StaticGaussianHead(cfg.Model.Gaussian_head)
        if initialize:
            state = torch.load(local_path(cfg.Model.dinov2_ckpt), map_location='cpu', weights_only=True)
            self.gaussian_head.image_backbone.load_state_dict(state, strict=True)
        self.gaussian_head.image_backbone.mask_token = None
        self.stage = 3
        self.set_stage(3)

    def set_stage(self, stage):
        if stage not in (1, 2, 3):
            raise ValueError(stage)
        self.stage = stage
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        if stage == 1:
            for name in SCALE_MODULES:
                getattr(self.gaussian_head, name).requires_grad_(True)
        else:
            self.gaussian_head.requires_grad_(True)
            for name in (*SCALE_MODULES, 'to_gaussians_sky', 'perceptual_loss'):
                getattr(self.gaussian_head, name).requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.geometry_model.eval()
        self.gaussian_head.perceptual_loss.eval()
        self.gaussian_head.to_gaussians_sky.eval()
        if self.stage == 1:
            self.gaussian_head.image_backbone.eval()
            self.gaussian_head.unet.eval()
        else:
            for name in SCALE_MODULES:
                getattr(self.gaussian_head, name).eval()
        return self

    def reconstruct(self, context, stage=3, supervision=None):
        if set(context) != {'image', 'intrinsics', 'extrinsics'}:
            raise ValueError('Only six RGB images and calibrated cameras may enter reconstruction')
        images = context['image']
        if images.shape[:3] != (1, 6, 3):
            raise ValueError(f'Expected [1,6,3,H,W], got {tuple(images.shape)}')
        if any(not torch.isfinite(context[k]).all() for k in ('intrinsics', 'extrinsics')):
            raise FloatingPointError('Camera projection requires finite input matrices')
        padded = pad_images(images)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('The reviewed Pi3 precision requires a GPU with BF16 support')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            geometry = self.geometry_model(padded)
        metric = supervision.get('input_metric_depth') if supervision else None
        with torch.autocast('cuda', enabled=False):
            return self.gaussian_head(geometry, padded.float(), context, stage=stage, metric_depth=metric)

    def forward(self, context, stage=3, supervision=None):
        return self.reconstruct(context, stage, supervision)

    def render(self, gaussians, cameras, image_shape):
        if set(cameras) != {'intrinsics', 'extrinsics'}:
            raise ValueError('Renderer accepts only target cameras')
        k, poses = cameras['intrinsics'][0], cameras['extrinsics'][0]
        if not torch.isfinite(k).all() or not torch.isfinite(poses).all():
            raise FloatingPointError('Rendering requires finite target camera matrices')
        h, w = image_shape
        views = len(k)
        if self.gaussian_head.renderer.resolution != [h, w]:
            raise ValueError('Renderer must use the full original image resolution')
        if not len(gaussians):
            zeros = gaussians.new_zeros((views, 1, h, w))
            return dict(image=zeros.expand(-1, 3, -1, -1), depth=zeros)
        def render_views(gs, camera_poses, camera_k):
            package = self.gaussian_head.renderer.render(
                gaussians=gs, c2w=camera_poses, K=camera_k,
                fovx=2 * torch.atan(w / (2 * camera_k[:, 0, 0])),
                fovy=2 * torch.atan(h / (2 * camera_k[:, 1, 1])),
                H=torch.full((len(camera_k),), h, device=k.device, dtype=torch.int32),
                W=torch.full((len(camera_k),), w, device=k.device, dtype=torch.int32),
                semantics=gs.new_zeros((len(gs), 1)))
            return {key: package[key] for key in ('image', 'depth')}

        if self.checkpoint_rendering and torch.is_grad_enabled() and gaussians.requires_grad:
            # Free each view's rasterizer work buffers before rendering the next.
            # Camera slices are explicit inputs, never a late-bound view closure.
            packages = [checkpoint(render_views, gaussians, poses[v:v + 1], k[v:v + 1], use_reentrant=False)
                        for v in range(views)]
            return {key: torch.cat([p[key] for p in packages]) for key in ('image', 'depth')}
        return render_views(gaussians, poses, k)


def parameter_counts(model):
    totals = dict(trainable=0, frozen=0, total=0)
    modules = {}
    disabled = {}
    for name, p in model.named_parameters():
        if '.perceptual_loss.' in name:
            continue
        n = p.numel()
        group = '.'.join(name.split('.')[:2])
        stat = modules.setdefault(group, dict(trainable=0, frozen=0, total=0))
        for counts in (totals, stat):
            counts['trainable' if p.requires_grad else 'frozen'] += n
            counts['total'] += n
        if 'to_gaussians_sky' in name or name.startswith(('geometry_model.conf_', 'geometry_model.camera_')):
            disabled[group] = disabled.get(group, 0) + n
    return dict(**totals, by_module=modules, disabled_or_unused=disabled, stage=model.stage,
                definition='Registered reconstruction parameters; excludes loss/metric networks; counts shared tensors once')
