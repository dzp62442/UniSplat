"""Three-stage static OmniScene training (see docs/OmniScene 数据集实验文档.md)."""
import argparse
import json

from omegaconf import OmegaConf

from omniscene.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment/omniscene_112x200.yaml')
    parser.add_argument('--pi3_ckpt')
    parser.add_argument('--dinov2_ckpt')
    parser.add_argument('--resume', help='Explicit checkpoint directory; by default auto-resume the latest in this run')
    parser.add_argument('--work_dir')
    parser.add_argument('--check-config', action='store_true', help='Resolve config without loading CUDA or weights')
    parser.add_argument('--check-data', action='store_true', help='Read one sample per stage/split; no model or notifications')
    parser.add_argument('overrides', nargs='*', help='OmegaConf key=value overrides')
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    for key in ('pi3_ckpt', 'dinov2_ckpt'):
        if getattr(args, key):
            cfg.Model[key] = getattr(args, key)
    if args.check_config:
        print(OmegaConf.to_yaml(cfg))
        return
    if args.check_data:
        from dataset.omniscene import OmniSceneDataset
        records = []
        for split, stage in [('train', 1), ('train', 2), ('train', 3), ('val', 3), ('mini', 3), ('total', 3)]:
            data = OmniSceneDataset(cfg.Dataset, split, stage)
            batch = data[0]
            records.append(dict(split=split, stage=stage, bins=len(data), token=batch['meta']['bin_token'],
                                context=list(batch['context']['image'].shape), target=list(batch['target']['image'].shape),
                                metric_depth='input_metric_depth' in batch['supervision'],
                                relative_depth='rel_depth' in batch['target'], mask='loss_mask' in batch['target']))
        print(json.dumps(records, indent=2))
        return
    from omniscene.training import run_training
    run_training(cfg, args.resume, args.work_dir)


if __name__ == '__main__':
    main()
