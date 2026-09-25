"""Explicit OmegaConf composition and the reviewed experiment contract."""
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=()):
    def read(file, stack):
        file = Path(file).resolve()
        if file in stack:
            raise ValueError(f"Config include cycle: {file}")
        own = OmegaConf.load(file)
        bases = [read(REPO_ROOT / p, stack + [file]) for p in own.pop('includes', [])]
        return OmegaConf.merge(*bases, own)

    path = Path(path)
    cfg = read(path if path.is_absolute() else REPO_ROOT / path, [])
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    h, w = cfg.Dataset.image_shape
    cfg.Model.encoder_image_shape = [h, ((w + 13) // 14) * 14]
    # The inherited head constructor takes width before height.
    cfg.Model.Gaussian_head.resolution = [w, h]
    return cfg


def validate_config(cfg):
    expected = {
        'Dataset.num_context_views': 6, 'Dataset.num_target_views': 18,
        'Dataset.train_batch_size': 1, 'Dataset.val_batch_size': 1,
        'Dataset.test_batch_size': 1, 'Dataset.use_dynamic_mask': True,
        'Dataset.load_lidar': False, 'Dataset.load_sky_mask': False,
        'Dataset.load_metric_confidence': False, 'Model.temporal_enabled': False,
        'Model.use_sky_branch': False, 'Model.pi3_decoder_size': 'large',
        'Model.Gaussian_head.dinov2_pretrain_img_size': 518,
        'Model.padding': 'right_replicate_to_14', 'Model.pi3_precision': 'bf16',
        'Loss.dyn_loss_weight': 0.0, 'Train.gradient_accumulation_steps': 1,
        'Train.max_steps': 100001, 'Train.val_every_steps': 1000,
        'Train.mini_test_every_n_val': 10, 'Train.final_mini_test': True,
        'Evaluation.pcc_reference': 'depth_anything_v2',
        'Evaluation.depth_semantics': 'accumulated_z',
        'Evaluation.pixel_protocol': 'full_image', 'Logging.backend': 'local',
        'Logging.wandb_mode': 'offline',
    }
    for name, value in expected.items():
        if OmegaConf.select(cfg, name) != value:
            raise ValueError(f"Reviewed protocol requires {name}={value!r}")
    if list(cfg.Train.stage_steps) != [44445, 33334, 22222]:
        raise ValueError('stage_steps must be [44445, 33334, 22222]')
    if list(cfg.Dataset.image_shape) not in ([112, 200], [224, 400]):
        raise ValueError('image_shape must be [112,200] or [224,400]')
    if list(cfg.Train.mini_test_stages) != [2, 3]:
        raise ValueError('Only stages 2 and 3 have periodic mini evaluations')
    if list(cfg.Evaluation.view_groups) != ['all_18', 'novel_12', 'input_6']:
        raise ValueError('Unexpected evaluation view groups')
    if list(cfg.Evaluation.metrics) != ['psnr', 'ssim', 'lpips', 'pcc']:
        raise ValueError('All four metrics are required')
    if cfg.Evaluation.split not in ('mini', 'total'):
        raise ValueError('Evaluation.split must be mini or total')
    if cfg.Dataset.num_workers < 0 or cfg.Loss.perceptual_chunk_size < 1:
        raise ValueError('Invalid worker/chunk count')


def config_identity(cfg):
    """Training identity excludes machine paths and notification settings."""
    obj = OmegaConf.to_container(cfg, resolve=True)
    for section in ('Feishu', 'Logging', 'Evaluation'):
        obj.pop(section, None)
    obj['Experiment'].pop('work_dir_root', None)
    for key in ('pi3_ckpt', 'dinov2_ckpt'):
        obj['Model'].pop(key, None)
    for key in ('vgg_ckpt', 'lpips_ckpt'):
        obj['Loss'].pop(key, None)
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def stage_for_step(completed_steps, stage_steps=(44445, 33334, 22222)):
    """Stage of the NEXT update, or 3 when training is finished."""
    if completed_steps < 0 or completed_steps > sum(stage_steps):
        raise ValueError(f'Invalid completed update count: {completed_steps}')
    end = 0
    for stage, count in enumerate(stage_steps, 1):
        end += count
        if completed_steps < end:
            return stage
    return len(stage_steps)


def due_events(step, cfg):
    if not 0 < step <= cfg.Train.max_steps:
        return False, False
    val = step % cfg.Train.val_every_steps == 0
    stage = stage_for_step(step - 1, cfg.Train.stage_steps)
    mini = (val and step // cfg.Train.val_every_steps % cfg.Train.mini_test_every_n_val == 0
            and stage in cfg.Train.mini_test_stages)
    return val, mini or (step == cfg.Train.max_steps and cfg.Train.final_mini_test)
