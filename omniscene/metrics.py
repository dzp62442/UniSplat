"""Full-image quality and DA V2 PCC, with bin-balanced 18/12/6 reporting."""
from collections import Counter
import math

import numpy as np
import torch
from skimage.metrics import structural_similarity

VIEW_GROUPS = {'all_18': slice(0, 18), 'novel_12': slice(0, 12), 'input_6': slice(12, 18)}
METRICS = ('psnr', 'ssim', 'lpips', 'pcc')


@torch.no_grad()
def compute_pcc(reference, predicted):
    if reference.shape != predicted.shape:
        raise ValueError('PCC tensors must have matching shapes')
    a, b = reference.reshape(-1).double(), predicted.reshape(-1).double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or a.numel() < 2:
        return predicted.new_tensor(float('nan'))
    a, b = a - a.mean(), b - b.mean()
    denom = a.square().sum().sqrt() * b.square().sum().sqrt()
    if denom <= 0:
        return predicted.new_tensor(float('nan'))
    return ((a * b).sum() / denom).clamp(-1, 1).float()


class ImageMetrics:
    def __init__(self, vgg_path, lpips_path, device):
        import lpips
        # pnet_rand avoids an implicit torchvision download; populate it strictly below.
        self.lpips = lpips.LPIPS(net='vgg', pnet_rand=True, model_path=str(lpips_path), verbose=False)
        weights = torch.load(vgg_path, map_location='cpu', weights_only=True)
        net_weights = {name: weights['features.' + name.split('.', 1)[1]]
                       for name in self.lpips.net.state_dict()}
        self.lpips.net.load_state_dict(net_weights, strict=True)
        self.lpips.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, gt, pred):
        if gt.shape != pred.shape or gt.shape[:2] != (18, 3):
            raise ValueError(f'Expected 18 matching RGB views, got {gt.shape}, {pred.shape}')
        mse = (gt.clamp(0, 1) - pred.clamp(0, 1)).square().mean((1, 2, 3))
        psnr = -10 * mse.log10()
        ssim = [structural_similarity(a.cpu().numpy(), b.cpu().numpy(), win_size=11,
                                      gaussian_weights=True, channel_axis=0, data_range=1.0)
                for a, b in zip(gt, pred)]
        perceptual = torch.cat([self.lpips(a[None], b[None], normalize=True).reshape(1)
                                for a, b in zip(gt, pred)])
        return dict(psnr=psnr, ssim=pred.new_tensor(ssim), lpips=perceptual)


def view_records(token, scene, image_metrics, reference, predicted, image_shape):
    if reference.shape != predicted.shape or reference.shape[0] != 18:
        raise ValueError(f'Expected matching 18-view depths: {token}')
    rows = []
    for group, indices in VIEW_GROUPS.items():
        row = dict(bin_token=token, scene_id=scene, view_group=group, height=image_shape[0], width=image_shape[1])
        row.update({name: value[indices].double().mean().item() for name, value in image_metrics.items()})
        row['pcc'] = compute_pcc(reference[indices], predicted[indices]).item()
        rows.append(row)
    return rows


def summarize_records(records, expected_tokens):
    expected = set(expected_tokens)
    if not expected or len(expected) != len(expected_tokens):
        raise ValueError('Expected a nonempty unique token list')
    if any(r['view_group'] not in VIEW_GROUPS for r in records):
        raise ValueError('Unknown metric group')
    summary = dict(expected_bins=len(expected), complete=True, primary_groups=['all_18', 'novel_12'],
                   pcc_reference='depth_anything_v2', depth_semantics='accumulated_z', pixel_protocol='full_image',
                   aggregation='Mean per-view RGB within bin, then equal-bin mean; PCC flattened per bin/group')
    for group in VIEW_GROUPS:
        rows = [r for r in records if r['view_group'] == group]
        counts = Counter(r['bin_token'] for r in rows)
        invalid = [dict(bin_token=r['bin_token'], metric=m) for r in rows for m in METRICS
                   if r.get(m) is None or not math.isfinite(r[m])]
        missing, extra = sorted(expected - counts.keys()), sorted(counts.keys() - expected)
        duplicates = sorted(k for k, n in counts.items() if n != 1)
        complete = not (missing or extra or duplicates or invalid)
        output = dict(num_bins=len(counts), num_records=len(rows), complete=complete,
                      missing_bins=missing, unexpected_bins=extra, duplicate_bins=duplicates, nonfinite_metrics=invalid)
        for name in METRICS:
            values = [r.get(name) for r in rows]
            output[name] = (math.fsum(values) / len(values)
                            if values and all(v is not None and math.isfinite(v) for v in values) else None)
        summary[group] = output
        summary['complete'] &= complete
    return summary


def summarize_times(rows):
    summary = dict(num_bins=len(rows), warmup_excluded=True,
                   reconstruction_boundary='Input H2D through final merged Gaussians; excludes target rendering/metrics/IO',
                   network_boundary='GPU-resident inputs through final merged Gaussians; includes CPU control work')
    for key in ('reconstruction_ms', 'network_reconstruction_ms', 'h2d_ms'):
        values = [r[key] for r in rows]
        summary[key] = (dict(mean=float(np.mean(values)), median=float(np.median(values)),
                             p95=float(np.percentile(values, 95))) if values else None)
    return summary
