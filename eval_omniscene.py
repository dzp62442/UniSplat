"""Evaluate a static UniSplat checkpoint on mini (2048) or total (30080) bins."""
import argparse
import json
from pathlib import Path

from omniscene.config import REPO_ROOT, load_config, config_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment/omniscene_112x200.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['mini', 'total'], default='total')
    parser.add_argument('--output-dir')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    from safetensors.torch import load_model
    import torch
    from dataset.omniscene import sha256_file
    from omniscene.model import StaticUniSplat
    from omniscene.evaluate import evaluate
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / 'model.safetensors'
    manifest = json.loads(checkpoint.with_name('manifest.json').read_text())
    if manifest['config_identity'] != config_identity(cfg):
        raise ValueError('Checkpoint does not match this experiment configuration')
    digest = sha256_file(checkpoint)
    if digest != manifest['model_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    if manifest['progress']['stage'] == 1:
        raise ValueError('Stage 1 has not trained the Gaussian branches')
    model = StaticUniSplat(cfg, initialize=False).cuda()
    load_model(model, str(checkpoint), strict=True)
    model.set_stage(3)
    model.eval()
    step = manifest['progress']['global_step']
    output = (Path(args.output_dir) if args.output_dir else
              REPO_ROOT / 'outputs' / cfg.Experiment.name / f'step_{step:06d}_{digest[:8]}' / args.split)
    summary = evaluate(model, cfg, output, dict(path=str(checkpoint), sha256=digest, global_step=step), args.split)
    print(json.dumps({k: summary[k] for k in ('complete', 'expected_bins', 'all_18', 'novel_12', 'timing')}, indent=2))
    if not summary['complete']:
        raise RuntimeError(f'Incomplete evaluation: {output}')


if __name__ == '__main__':
    main()
