"""UniSplat losses and coefficients, with SVF-GS floating mask semantics."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def masked_rgb_mse(predicted, target, mask):
    return ((predicted * mask[:, None] - target * mask[:, None]) ** 2).mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, cfg, perceptual=None):
        super().__init__()
        self.cfg = cfg.Loss
        self.register_buffer('bounds', torch.tensor(cfg.Model.Gaussian_head.pts_range, dtype=torch.float32))
        if perceptual is None:
            from model.loss_func.percept_loss import LPIPS
            from .model import local_path
            perceptual = LPIPS(vgg_ckpt=str(local_path(self.cfg.vgg_ckpt)),
                               lpips_ckpt=str(local_path(self.cfg.lpips_ckpt)))
        self.perceptual = perceptual.eval().requires_grad_(False)

    def forward(self, model, reconstruction, target, stage):
        if stage == 1:
            scale, shift, valid = reconstruction['alignment']
            if not valid.any():
                raise ValueError('No valid depth alignments')
            losses = {
                'scale': (reconstruction['pred_scale'][valid] - scale[valid]).abs().mean() * self.cfg.scale_weight,
                'shift': (reconstruction['pred_shift'][valid] - shift[valid]).abs().mean() * self.cfg.shift_weight,
            }
            return losses
        cameras = {key: target[key] for key in ('intrinsics', 'extrinsics')}
        gt, mask = target['image'][0], target['loss_mask'][0]
        rendered = model.render(reconstruction['gaussians'], cameras, gt.shape[-2:])
        pred = rendered['image']
        rec = masked_rgb_mse(pred, gt, mask) * self.cfg.rec_weight * self.cfg.rec_view_weight
        perceptual = pred.new_zeros(())
        for start in range(0, len(gt), self.cfg.perceptual_chunk_size):
            end = start + self.cfg.perceptual_chunk_size
            m = mask[start:end, None]
            predicted, truth = pred[start:end] * m, gt[start:end] * m
            # Eighteen VGG graphs otherwise dominate memory at 224x400. Recompute
            # only the frozen, eval-mode loss network; the objective is unchanged.
            if self.cfg.checkpoint_perceptual and torch.is_grad_enabled():
                values = checkpoint(self.perceptual, predicted, truth, use_reentrant=False)
            else:
                values = self.perceptual(predicted, truth)
            perceptual = perceptual + values.sum() / len(gt)
        perceptual = perceptual * self.cfg.perceptual_weight * self.cfg.perceptual_view_weight
        input_cameras = {k: v[:, 12:] for k, v in cameras.items()}
        voxel = model.render(reconstruction['voxel_gaussians'], input_cameras, gt.shape[-2:])['image']
        points = reconstruction['anchor_points'][0]
        inside = ((points > self.bounds[:3]) & (points < self.bounds[3:])).all(-1)
        aux = ((voxel - gt[12:]) ** 2 * inside[:, None]).mean() * self.cfg.aux_voxel_weight
        return dict(rec=rec, perceptual=perceptual, aux_voxel=aux)
