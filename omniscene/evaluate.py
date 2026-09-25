"""One reconstruction per bin, full target rendering, metrics and timings."""
import csv
import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import config_identity
from .io import capture_rng, restore_rng, isolated_evaluation, to_device, write_json
from .metrics import ImageMetrics, view_records, summarize_records, summarize_times
from .model import local_path, parameter_counts
from dataset.omniscene import OmniSceneDataset, sha256_file


def make_loader(dataset, cfg, sampler=None):
    generator = torch.Generator().manual_seed(cfg.Experiment.seed)
    return DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False,
                      num_workers=cfg.Dataset.num_workers, pin_memory=cfg.Dataset.pin_memory,
                      generator=generator)


def evaluation_identity(cfg, dataset, checkpoint, split):
    return dict(config_identity=config_identity(cfg), split=split, checkpoint=checkpoint,
                    manifest_sha256=dataset.manifest_sha256, tokens_sha256=dataset.tokens_sha256,
                    expected_bins=len(dataset), image_shape=list(cfg.Dataset.image_shape),
                    encoder_image_shape=list(cfg.Model.encoder_image_shape), temporal_enabled=False,
                    warmup_steps=cfg.Evaluation.warmup_steps, save_images=cfg.Evaluation.save_images,
                    metric_weights_sha256={name: sha256_file(local_path(cfg.Loss[name]))
                                           for name in ('vgg_ckpt', 'lpips_ckpt')},
                    pcc_reference='depth_anything_v2', depth_semantics='accumulated_z', pixel_protocol='full_image')


def load_cached_evaluation(output_dir, identity, expected_tokens):
    """Validate completed artifacts without allocating the reconstruction model."""
    output_dir = Path(output_dir)
    provenance_file = output_dir / 'data_provenance.json'
    if not provenance_file.is_file():
        return None
    if json.loads(provenance_file.read_text()) != identity:
        raise ValueError(f'Refusing to mix evaluation identities: {output_dir}')
    try:
        previous = json.loads((output_dir / 'evaluation_summary.json').read_text())
        if previous.get('complete') and all((output_dir / f).is_file() for f in
                                            ('per_bin_metrics.csv', 'reconstruction_time.json', 'model_parameters.json')):
            with (output_dir / 'per_bin_metrics.csv').open() as f:
                stored = list(csv.DictReader(f))
            for row in stored:
                for name in ('psnr', 'ssim', 'lpips', 'pcc'):
                    row[name] = float(row[name])
            recomputed = summarize_records(stored, expected_tokens)
            times = json.loads((output_dir / 'reconstruction_time.json').read_text())
            same_metrics = all(previous.get(group) == recomputed[group] for group in ('all_18', 'novel_12', 'input_6'))
            valid_times = ([r['bin_token'] for r in times.get('per_bin', [])] == expected_tokens
                           and times.get('num_bins') == len(expected_tokens))
            same_checkpoint = all(r.get('checkpoint_sha256') == identity['checkpoint']['sha256']
                                  and int(r['global_step']) == identity['checkpoint']['global_step'] for r in stored)
            if recomputed['complete'] and same_metrics and valid_times and same_checkpoint:
                return previous
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        # Partial/malformed result files must be recomputed, never called complete.
        return None
    return None


def evaluate(model, cfg, output_dir, checkpoint, split='mini', metrics=None, logger=None):
    dataset = OmniSceneDataset(cfg.Dataset, split, stage=3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    identity = evaluation_identity(cfg, dataset, checkpoint, split)
    previous = load_cached_evaluation(output_dir, identity, dataset.bin_tokens)
    # Existing outputs without provenance are not a verifiable cache.
    write_json(output_dir / 'data_provenance.json', identity)
    summary_file = output_dir / 'evaluation_summary.json'
    if previous is not None:
        return previous
    if metrics is None:
        rng = capture_rng()
        try:
            metrics = ImageMetrics(local_path(cfg.Loss.vgg_ckpt), local_path(cfg.Loss.lpips_ckpt), device)
        finally:
            restore_rng(rng)
    rows, timing_rows = [], []
    error_text = None
    write_json(output_dir / 'model_parameters.json', parameter_counts(model))
    fields = ['bin_token', 'scene_id', 'view_group', 'height', 'width', 'psnr', 'ssim', 'lpips', 'pcc',
              'split', 'global_step', 'checkpoint_sha256']
    try:
        with isolated_evaluation(model), (output_dir / 'per_bin_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            loader = make_loader(dataset, cfg)
            warmup = next(iter(loader))
            warm_context = to_device(warmup['context'], device)
            for _ in range(cfg.Evaluation.warmup_steps):
                warm_result = model.reconstruct(warm_context, stage=3)
                del warm_result
            del warm_context, warmup
            torch.cuda.synchronize(device)
            for index, batch in enumerate(loader):
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                context = to_device(batch['context'], device)
                torch.cuda.synchronize(device)
                after_transfer = time.perf_counter()
                reconstruction = model.reconstruct(context, stage=3)
                torch.cuda.synchronize(device)
                finished = time.perf_counter()
                token = batch['meta']['bin_token'][0]
                timing_rows.append(dict(bin_token=token, reconstruction_ms=(finished - begin) * 1000,
                                        network_reconstruction_ms=(finished - after_transfer) * 1000,
                                        h2d_ms=(after_transfer - begin) * 1000))
                target = to_device(batch['target'], device)
                rendered = model.render(reconstruction['gaussians'],
                                        {k: target[k] for k in ('intrinsics', 'extrinsics')}, cfg.Dataset.image_shape)
                scores = metrics(target['image'][0], rendered['image'])
                records = view_records(token, batch['meta']['scene_id'][0], scores,
                                       target['rel_depth'][0], rendered['depth'][:, 0], cfg.Dataset.image_shape)
                for row in records:
                    row.update(split=split, global_step=checkpoint['global_step'],
                               checkpoint_sha256=checkpoint['sha256'])
                rows.extend(records)
                writer.writerows(records)
                stream.flush()
                if cfg.Evaluation.save_images:
                    from PIL import Image
                    directory = output_dir / 'images' / token
                    directory.mkdir(parents=True, exist_ok=True)
                    for view, rgb in enumerate(rendered['image']):
                        image = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                        Image.fromarray(image).save(directory / f'{view:02d}.png')
                if logger and (index + 1) % 100 == 0:
                    logger.info('Evaluation %s: %d/%d bins', split, index + 1, len(dataset))
                del reconstruction, rendered, scores, target, context
    except BaseException as error:
        error_text = f'{type(error).__name__}: {error}'
        raise
    finally:
        summary = summarize_records(rows, dataset.bin_tokens)
        summary.update(checkpoint=checkpoint, split=split, image_shape=list(cfg.Dataset.image_shape), error=error_text)
        if error_text is not None:
            summary['complete'] = False
        timing_summary = summarize_times(timing_rows)
        timing_summary.update(per_bin=timing_rows, gpu=torch.cuda.get_device_name(device),
                              torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                              pi3_precision=cfg.Model.pi3_precision, head_precision='float32',
                              rope_backend=type(model.gaussian_head.rope).__module__,
                              image_shape=list(cfg.Dataset.image_shape),
                              encoder_image_shape=list(cfg.Model.encoder_image_shape))
        summary['timing'] = {k: v for k, v in timing_summary.items() if k != 'per_bin'}
        write_json(output_dir / 'reconstruction_time.json', timing_summary)
        write_json(summary_file, summary)
    return summary


def validate(model, cfg, criterion, logger=None):
    dataset = OmniSceneDataset(cfg.Dataset, 'val', stage=model.stage)
    device = next(model.parameters()).device
    totals = {}
    with isolated_evaluation(model):
        for batch in make_loader(dataset, cfg):
            context, target = to_device(batch['context'], device), to_device(batch['target'], device)
            supervision = to_device(batch['supervision'], device)
            reconstruction = model(context, stage=model.stage, supervision=supervision)
            losses = criterion(model, reconstruction, target, model.stage)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.item() / len(dataset)
    output = dict(stage=model.stage, num_bins=len(dataset), losses=totals,
                  geometry='metric3d_aligned' if model.stage in (1, 2) else 'predicted_scale_shift')
    if logger:
        logger.info('Validation: %s', output)
    return output
