"""Exact optimizer-step budgets with stage-aware freezing and recoverable evaluations."""
import json
import logging
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Sampler

from .checkpoint import find_latest_checkpoint, save_checkpoint, load_training_state, restore_checkpoint, prune_checkpoints
from .config import REPO_ROOT, config_identity, due_events, stage_for_step
from .evaluate import make_loader, evaluate, validate, evaluation_identity, load_cached_evaluation
from .io import to_device, write_json, restore_rng
from .losses import ReconstructionLoss
from .model import StaticUniSplat, parameter_counts, local_path
from .notify import FeishuNotifier
from dataset.omniscene import OmniSceneDataset, sha256_file


class ResumeSampler(Sampler):
    """Prefetch never advances the acknowledged training cursor."""
    def __init__(self, length, seed, epoch, offset):
        self.length, self.seed, self.epoch, self.offset = length, seed, epoch, offset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.length, generator=generator).tolist()[self.offset:])

    def __len__(self):
        return self.length - self.offset


def build_optimizer(model, cfg, stage):
    model.set_stage(stage)
    rates = cfg.Optimizer.stages[stage - 1]
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = rates.image_backbone_lr if name.startswith('gaussian_head.image_backbone.') else rates.head_lr
        if lr is None:
            raise ValueError(f'No learning rate for {name} at stage {stage}')
        wd = 0.0 if param.ndim == 1 or name.endswith('.bias') else cfg.Optimizer.weight_decay
        groups.setdefault((float(lr), wd), []).append(param)
    optimizer = torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=wd) for (lr, wd), ps in groups.items()],
                                  betas=tuple(cfg.Optimizer.betas), eps=cfg.Optimizer.eps, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g['lr'] for g in optimizer.param_groups], total_steps=cfg.Train.stage_steps[stage - 1],
        **OmegaConf.to_container(cfg.Scheduler))
    return optimizer, scheduler


def get_logger(directory):
    logger = logging.getLogger('unisplat.omniscene')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / 'train.log')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    return logger


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for UniSplat sparse convolutions and rendering')


def run_training(cfg, resume=None, work_dir=None):
    if int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This reviewed experiment requires one GPU and effective batch size 1')
    directory = (Path(work_dir).resolve() if work_dir else
                 (REPO_ROOT / cfg.Experiment.work_dir_root / cfg.Experiment.name).resolve())
    directory.mkdir(parents=True, exist_ok=True)
    logger = get_logger(directory)
    automatic = resume is None
    resume = find_latest_checkpoint(directory) if automatic else Path(resume).expanduser().resolve()
    if resume is not None:
        logger.info('%s checkpoint: %s', 'Auto-resuming from' if automatic else 'Explicit resume from', resume)
    else:
        logger.info('No previous checkpoint in %s; starting from base weights', directory)
    data = OmniSceneDataset(cfg.Dataset, 'train', stage=1)
    data_identity = dict(manifest_sha256=data.manifest_sha256, tokens_sha256=data.tokens_sha256, num_bins=len(data))
    state = None
    resume_rng = None
    manifest = None
    if resume:
        state, manifest = load_training_state(resume, cfg, data_identity)
        logger.info('Restoring optimizer update %d/%d, stage %d, stage update %d',
                    state['progress']['global_step'], cfg.Train.max_steps,
                    state['progress']['stage'], state['progress']['stage_step'])
    elif (directory / 'training_complete.json').exists():
        raise FileNotFoundError('Run has a completion marker but no checkpoint; use a new --work_dir to restart')
    progress = (state['progress'].copy() if state else
                dict(global_step=0, stage=1, stage_step=0, epoch=0, offset=0,
                     val_count=0, last_validation_step=0, last_mini_test_step=0))
    event_path = directory / 'events.json'
    if event_path.exists() and json.loads(event_path.read_text())['global_step'] > progress['global_step']:
        raise ValueError('This output directory contains later results; resume latest or use a new directory')
    completion_file = directory / 'training_complete.json'
    if resume and progress['global_step'] == cfg.Train.max_steps and completion_file.is_file():
        completion = json.loads(completion_file.read_text())
        if completion.get('global_step') == cfg.Train.max_steps and completion.get('last_mini_test_step') == cfg.Train.max_steps:
            final_data = OmniSceneDataset(cfg.Dataset, 'mini', stage=3)
            checkpoint_identity = dict(path=str(resume / 'model.safetensors'), sha256=manifest['model_sha256'],
                                       global_step=progress['global_step'])
            identity = evaluation_identity(cfg, final_data, checkpoint_identity, 'mini')
            final_dir = directory / 'evaluation' / f'step_{progress["global_step"]:06d}' / 'mini'
            if load_cached_evaluation(final_dir, identity, final_data.bin_tokens) is not None:
                logger.info('Training and final mini evaluation already complete: %s; nothing to resume', directory)
                return
    require_cuda()
    torch.cuda.set_device(0)
    random.seed(cfg.Experiment.seed)
    np.random.seed(cfg.Experiment.seed)
    torch.manual_seed(cfg.Experiment.seed)
    torch.cuda.manual_seed_all(cfg.Experiment.seed)
    model = StaticUniSplat(cfg, initialize=state is None).cuda()
    optimizer, scheduler = build_optimizer(model, cfg, progress['stage'])
    scaler = torch.amp.GradScaler('cuda')
    if state is not None:
        restore_checkpoint(resume, model, optimizer, scheduler, scaler, state)
        resume_rng = state['rng']
        del state
    criterion = ReconstructionLoss(cfg).cuda().eval()
    OmegaConf.save(cfg, directory / 'resolved_config.yaml')
    record = dict(config_identity=config_identity(cfg), data_identity=data_identity,
                  git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
                  reference_revisions=dict(SVF_GS='af39b31', depthsplat='405b9a5'),
                  source_weights={k: dict(path=str(local_path(p)), sha256=sha256_file(local_path(p)))
                                  for k, p in [('pi3', cfg.Model.pi3_ckpt), ('dinov2', cfg.Model.dinov2_ckpt),
                                               ('vgg', cfg.Loss.vgg_ckpt), ('lpips', cfg.Loss.lpips_ckpt)]})
    prior_manifest = Path(resume).resolve().parents[1] / 'run_manifest.json' if resume else directory / 'run_manifest.json'
    if prior_manifest.exists():
        prior = json.loads(prior_manifest.read_text())
        for name in ('vgg', 'lpips'):
            if prior['source_weights'][name]['sha256'] != record['source_weights'][name]['sha256']:
                raise ValueError(f'Resume changed the loss-network weights: {name}')
    record['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True))
    write_json(directory / 'run_manifest.json', record)
    notifier = FeishuNotifier(cfg.Feishu, directory)
    notifier.emit('train_start', progress['global_step'], f'UniSplat 训练开始：{cfg.Experiment.name}',
                  json.dumps(dict(resolution=list(cfg.Dataset.image_shape), protocol='static_six_views',
                                  stages=list(cfg.Train.stage_steps), progress=progress, work_dir=str(directory),
                                  initialization=record['source_weights'], parameters=parameter_counts(model)),
                             ensure_ascii=False, indent=2))
    stage_ends = list(np.cumsum(list(cfg.Train.stage_steps)))
    started = time.monotonic()
    start_step = progress['global_step']
    checkpoint = Path(resume).resolve() if resume else None

    def save_events():
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        write_json(event_path, dict(**progress, checkpoint_sha256=manifest['model_sha256']))

    def events():
        nonlocal checkpoint
        step = progress['global_step']
        val_due, mini_due = due_events(step, cfg)
        if event_path.exists():
            completed = json.loads(event_path.read_text())
            if completed['global_step'] == step:
                manifest = json.loads((checkpoint / 'manifest.json').read_text())
                if completed['checkpoint_sha256'] != manifest['model_sha256']:
                    raise ValueError('Evaluation journal belongs to different checkpoint weights')
                for key in ('last_validation_step', 'last_mini_test_step', 'val_count'):
                    progress[key] = max(progress[key], completed[key])
        if val_due and progress['last_validation_step'] < step:
            validation = validate(model, cfg, criterion, logger)
            write_json(directory / 'validation' / f'step_{step:06d}.json', validation)
            progress['last_validation_step'] = step
            progress['val_count'] = step // cfg.Train.val_every_steps
            save_events()
        if mini_due:
            if checkpoint is None:
                raise RuntimeError('Mini evaluation requires a saved checkpoint')
            manifest = json.loads((checkpoint / 'manifest.json').read_text())
            identity = dict(path=str(checkpoint / 'model.safetensors'), sha256=manifest['model_sha256'], global_step=step)
            result_dir = directory / 'evaluation' / f'step_{step:06d}' / 'mini'
            # Reusing an exact complete evaluation also repairs a missing notification after a crash.
            summary = evaluate(model, cfg, result_dir, identity, logger=logger)
            elapsed = time.monotonic() - started
            completed_steps = max(step - start_step, 1)
            notification = dict(step=step, work_dir=str(directory), split='mini', results=str(result_dir),
                                complete=summary['complete'], expected_bins=summary['expected_bins'],
                                all_18=summary['all_18'], novel_12=summary['novel_12'], timing=summary['timing'],
                                eta_seconds=elapsed / completed_steps * (cfg.Train.max_steps - step))
            notifier.emit('mini_test_complete', step, f'UniSplat mini 评估：{cfg.Experiment.name} / {step}',
                          json.dumps(notification, ensure_ascii=False, indent=2))
            if not summary['complete']:
                raise RuntimeError(f'Incomplete mini evaluation at {step}; inspect {result_dir}')
            progress['last_mini_test_step'] = step
            save_events()

    skipped = 0
    iterator = None
    active_stage = None
    try:
        if resume_rng is not None:
            restore_rng(resume_rng)
        if resume:
            events()
        while progress['global_step'] < cfg.Train.max_steps:
            stage = stage_for_step(progress['global_step'], cfg.Train.stage_steps)
            if stage != progress['stage']:
                optimizer, scheduler = build_optimizer(model, cfg, stage)
                progress.update(stage=stage, stage_step=0)
            if active_stage != stage or iterator is None:
                data = OmniSceneDataset(cfg.Dataset, 'train', stage=stage)
                sampler = ResumeSampler(len(data), cfg.Experiment.seed, progress['epoch'], progress['offset'])
                iterator = iter(make_loader(data, cfg, sampler))
                active_stage = stage
                counts = parameter_counts(model)
                write_json(directory / f'model_parameters_stage{stage}.json', counts)
                logger.info('Stage %s, parameters: %s', stage, {k: counts[k] for k in ('trainable', 'frozen', 'total')})
            try:
                batch = next(iterator)
            except StopIteration:
                progress['epoch'] += 1
                progress['offset'] = 0
                iterator = None
                continue
            progress['offset'] += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = to_device(batch, 'cuda')
            reconstruction = model(batch['context'], stage=stage, supervision=batch['supervision'])
            losses = criterion(model, reconstruction, batch['target'], stage)
            total = sum(losses.values())
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite loss at step {progress["global_step"]}: {batch["meta"]["bin_token"]}')
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), cfg.Train.grad_max_norm)
            if not torch.isfinite(norm):
                scaler.update(new_scale=scaler.get_scale() / 2)
                skipped += 1
                logger.warning('Skipped nonfinite gradient update (%d); optimizer step budget unchanged', skipped)
                if skipped >= cfg.Train.max_consecutive_skipped_updates:
                    raise FloatingPointError('Too many consecutive invalid updates')
                del total, losses, reconstruction, batch
                continue
            scaler.step(optimizer)
            scaler.update()
            skipped = 0
            scheduler.step()
            progress['global_step'] += 1
            progress['stage_step'] += 1
            step = progress['global_step']
            if step % cfg.Train.log_every_steps == 0 or step == 1:
                metrics = {k: v.item() for k, v in losses.items()}
                alignment = reconstruction.get('alignment')
                metrics['valid_alignment_cameras'] = int(alignment[2].sum()) if alignment is not None else None
                logger.info('Step %d/%d stage=%d loss=%s lr=%s', step, cfg.Train.max_steps, stage,
                            metrics, [g['lr'] for g in optimizer.param_groups])
                with (directory / 'train.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step, stage=stage, losses=metrics)) + '\n')
            del total, losses, reconstruction, batch
            val_due, mini_due = due_events(step, cfg)
            if val_due or mini_due or step in stage_ends:
                checkpoint = save_checkpoint(directory / 'checkpoints', model, optimizer, scheduler, scaler,
                                             progress, cfg, data_identity)
                events()
                prune_checkpoints(directory / 'checkpoints', stage_ends)
        if progress['last_mini_test_step'] != cfg.Train.max_steps:
            raise RuntimeError('Training updates finished, but final mini evaluation is not complete')
        write_json(directory / 'training_complete.json', progress)
        logger.info('All %d updates and final mini evaluation completed', cfg.Train.max_steps)
    finally:
        notifier.close()
