"""Atomic experiment records and RNG preservation."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w') as f:
        json.dump(clean_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@contextmanager
def isolated_evaluation(model):
    rng, training = capture_rng(), model.training
    try:
        model.eval()
        # Some upstream RoPE modules cache tensors on their first call. no_grad
        # keeps those caches usable in a subsequent training backward after resume.
        with torch.no_grad():
            yield
    finally:
        model.train(training)
        restore_rng(rng)


def to_device(tree, device):
    if torch.is_tensor(tree):
        return tree.to(device, non_blocking=True)
    if isinstance(tree, dict):
        return {k: to_device(v, device) for k, v in tree.items()}
    return tree
