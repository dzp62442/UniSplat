"""Bounded real-data GPU checks; no training run, checkpoint, or notifications."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import default_collate

from dataset.omniscene import OmniSceneDataset
from omniscene.config import load_config
from omniscene.io import to_device, write_json
from omniscene.losses import ReconstructionLoss
from omniscene.metrics import ImageMetrics, view_records
from omniscene.model import StaticUniSplat, local_path, parameter_counts
from omniscene.training import build_optimizer


def renderer_contract(model, shape):
    h, w = shape
    gs = torch.tensor([[0., 0., 5., 1., 1., 1., .5, 1., 0., 0., 0., .1, .1, .1]], device='cuda')
    k = torch.tensor([[180., 0., w * .43], [0., 180., h * .46], [0., 0., 1.]], device='cuda')
    ks = k[None, None].repeat(1, 3, 1, 1)
    ks[0, 1, 0, 2] += 5
    ks[0, 2, 0, 2] -= 5
    cameras = dict(intrinsics=ks, extrinsics=torch.eye(4, device='cuda')[None, None].repeat(1, 3, 1, 1))
    results, grads = [], []
    for enabled in (False, True):
        model.checkpoint_rendering = enabled
        leaf = gs.clone().requires_grad_()
        rendered = model.render(leaf, cameras, shape)
        (rendered['image'].square().mean() + rendered['depth'].mean() * .01).backward()
        grads.append(leaf.grad)
        results.append({k: v.detach() for k, v in rendered.items()})
    model.checkpoint_rendering = True
    torch.testing.assert_close(grads[0], grads[1])
    for key in ('image', 'depth'):
        torch.testing.assert_close(results[0][key], results[1][key])
    alpha = results[0]['image'][:, :1]
    torch.testing.assert_close(results[0]['depth'], 5 * alpha, rtol=1e-5, atol=1e-6)
    assert alpha.max() < .51  # Accumulated Z, not alpha-normalized expected Z.
    for view in range(3):
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        a = alpha[view, 0]
        cx, cy = (xx * a).sum() / a.sum(), (yy * a).sum() / a.sum()
        torch.testing.assert_close(cx, ks[0, view, 0, 2] - .5, atol=.1, rtol=0)
        torch.testing.assert_close(cy, ks[0, view, 1, 2] - .5, atol=.1, rtol=0)
    return dict(accumulated_camera_z=True, principal_point=True, checkpoint_gradient_parity=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment/omniscene_112x200.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--optimizer-step', action='store_true', help='One disposable in-memory update per stage')
    args = parser.parse_args()
    cfg = load_config(args.config, ['Feishu.enabled=false', 'Dataset.num_workers=0'])
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    model = StaticUniSplat(cfg).cuda()
    criterion = ReconstructionLoss(cfg).cuda().eval()
    report = dict(config=args.config, gpu=torch.cuda.get_device_name(), stages=[], complete=False)
    try:
        for stage in args.stages:
            print(f'Checking stage {stage}', flush=True)
            model.set_stage(stage)
            if args.optimizer_step:
                optimizer, scheduler = build_optimizer(model, cfg, stage)
            model.train()
            model.zero_grad(set_to_none=True)
            data = OmniSceneDataset(cfg.Dataset, 'val', stage)
            batch = to_device(default_collate([data[0]]), 'cuda')
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = model(batch['context'], stage=stage, supervision=batch['supervision'])
            losses = criterion(model, result, batch['target'], stage)
            loss = sum(losses.values())
            assert torch.isfinite(loss), losses
            loss.backward()
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            assert not bad, bad
            assert all(p.grad is None for p in model.geometry_model.parameters())
            assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
            if args.optimizer_step:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), cfg.Train.grad_max_norm)
                optimizer.step()
                scheduler.step()
            record = dict(stage=stage, losses={k: v.item() for k, v in losses.items()},
                          parameters=parameter_counts(model), peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                          elapsed_seconds=time.perf_counter() - start)
            record['disposable_optimizer_update'] = args.optimizer_step
            if stage > 1:
                h, w = cfg.Dataset.image_shape
                assert result['num_point_gaussians'] == 6 * h * w
                assert tuple(result['anchor_points'].shape) == (1, 6, h, w, 3)
                assert torch.isfinite(result['gaussians']).all()
                assert not hasattr(model.gaussian_head, 'history_queue')
                record.update(num_point_gaussians=result['num_point_gaussians'],
                              num_voxel_gaussians=result['num_voxel_gaussians'])
            report['stages'].append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'parameters'}), flush=True)
            del batch, result, losses, loss
            model.zero_grad(set_to_none=True)
            if args.optimizer_step:
                del optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        if args.metrics:
            model.set_stage(3)
            model.eval()
            data = OmniSceneDataset(cfg.Dataset, 'mini', 3)
            batch = to_device(default_collate([data[0]]), 'cuda')
            metric = ImageMetrics(local_path(cfg.Loss.vgg_ckpt), local_path(cfg.Loss.lpips_ckpt), 'cuda')
            with torch.no_grad():
                a = model.reconstruct(batch['context'])
                other = to_device(default_collate([data[1]])['context'], 'cuda')
                other_result = model.reconstruct(other)
                del other, other_result
                # Intervening different bins cannot affect reconstruction.
                b = model.reconstruct(batch['context'])
                torch.testing.assert_close(a['gaussians'], b['gaussians'])
                camera = {k: batch['target'][k] for k in ('intrinsics', 'extrinsics')}
                rendered = model.render(a['gaussians'], camera, cfg.Dataset.image_shape)
                scores = metric(batch['target']['image'][0], rendered['image'])
                report['metric_smoke_only'] = view_records(batch['meta']['bin_token'][0], batch['meta']['scene_id'][0],
                                                         scores, batch['target']['rel_depth'][0], rendered['depth'][:, 0],
                                                         cfg.Dataset.image_shape)
                assert all(torch.isfinite(s).all() for s in scores.values())
            del a, b, rendered, scores, batch, metric
            from omniscene.evaluate import evaluate
            limited = OmniSceneDataset(cfg.Dataset, 'mini', 3)
            limited.bin_tokens = limited.bin_tokens[:2]
            import hashlib
            limited.tokens_sha256 = hashlib.sha256(json.dumps(limited.bin_tokens).encode()).hexdigest()
            output = Path(args.output).with_suffix('') / 'evaluation_smoke' / f'run_{time.time_ns()}'
            # Exercise the production writer/timer; this explicitly limited fixture is not a mini result.
            with patch('omniscene.evaluate.OmniSceneDataset', return_value=limited):
                summary = evaluate(model, cfg, output, dict(path='untrained-smoke-only', sha256='smoke-only', global_step=0))
            assert summary['complete'], summary
            assert summary['timing']['num_bins'] == 2
            report['evaluation_pipeline_smoke'] = dict(num_bins=2, complete=True, output=str(output))
            print('Metric and stateless reconstruction checks passed', flush=True)
        report['renderer_contract'] = renderer_contract(model, cfg.Dataset.image_shape)
        report['complete'] = True
    finally:
        write_json(args.output, report)


if __name__ == '__main__':
    main()
