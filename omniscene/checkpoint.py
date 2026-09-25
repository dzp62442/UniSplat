"""Step-based resume state, committed alongside an immutable model snapshot."""
import os
from pathlib import Path
import shutil
import uuid

import torch
from safetensors.torch import save_model, load_model

from .io import capture_rng, restore_rng, write_json
from .config import config_identity
from dataset.omniscene import sha256_file


def find_latest_checkpoint(directory):
    """Find the last committed snapshot of this run, independent of the latest link.

    Interrupted atomic writes have a .tmp suffix and are never candidates. A
    damaged committed snapshot is an error, not permission to silently start over.
    """
    root = Path(directory) / 'checkpoints'
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith('step_')
                  and p.name[5:].isdigit()]
    if not candidates:
        if (root / 'latest').exists() or (root / 'latest').is_symlink():
            raise FileNotFoundError(f'No committed step_* checkpoint in {root}; inspect latest or specify --resume')
        return None
    newest = max(candidates, key=lambda p: int(p.name[5:]))
    missing = [name for name in ('manifest.json', 'model.safetensors', 'training_state.pth')
               if not (newest / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Newest checkpoint is incomplete: {newest}; missing {missing}. '
                                'Refusing to restart or overwrite this run.')
    return newest.resolve()


def save_checkpoint(root, model, optimizer, scheduler, scaler, progress, cfg, data_identity):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f'step_{progress["global_step"]:06d}'
    if dest.exists():
        raise FileExistsError(f'Checkpoint already exists: {dest}; use --resume')
    temporary = root / (dest.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.mkdir()
    try:
        save_model(model, str(temporary / 'model.safetensors'))
        state = dict(progress=progress.copy(), config_identity=config_identity(cfg), data_identity=data_identity,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     rng=capture_rng())
        torch.save(state, temporary / 'training_state.pth')
        metadata = dict(progress=progress.copy(), config_identity=config_identity(cfg), data_identity=data_identity,
                        model_sha256=sha256_file(temporary / 'model.safetensors'),
                        state_sha256=sha256_file(temporary / 'training_state.pth'))
        write_json(temporary / 'manifest.json', metadata)
        temporary.rename(dest)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    link = root / ('.latest-' + uuid.uuid4().hex)
    link.symlink_to(dest.name, target_is_directory=True)
    os.replace(link, root / 'latest')
    return dest


def load_training_state(path, cfg, data_identity):
    import json
    path = Path(path).resolve()
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['config_identity'] != config_identity(cfg) or manifest['data_identity'] != data_identity:
        raise ValueError('Checkpoint configuration or data manifest differs from this run')
    if sha256_file(path / 'model.safetensors') != manifest['model_sha256']:
        raise ValueError('Checkpoint model hash mismatch')
    if sha256_file(path / 'training_state.pth') != manifest['state_sha256']:
        raise ValueError('Checkpoint training state hash mismatch')
    state = torch.load(path / 'training_state.pth', map_location='cpu', weights_only=False)
    if (state['progress'] != manifest['progress'] or state['config_identity'] != manifest['config_identity']
            or state['data_identity'] != manifest['data_identity']):
        raise ValueError('Checkpoint weights and resume state are not paired')
    return state, manifest


def restore_checkpoint(path, model, optimizer, scheduler, scaler, state):
    load_model(model, str(Path(path) / 'model.safetensors'), strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'])


def prune_checkpoints(root, stage_ends, keep_last=2):
    paths = sorted(p for p in Path(root).glob('step_*') if p.is_dir() and p.name[5:].isdigit())
    protected = set(paths[-keep_last:])
    for path in paths:
        step = int(path.name[5:])
        # Keep stage boundaries and every checkpoint used for periodic evaluation.
        if path not in protected and step not in stage_ends and not (step >= 50000 and step % 10000 == 0):
            shutil.rmtree(path)
