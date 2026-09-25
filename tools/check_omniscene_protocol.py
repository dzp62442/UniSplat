"""Compare actual assets against local reference loader functions, without CUDA."""
import argparse
import ast
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import PIL
from PIL import Image
import torch

from dataset.omniscene import CAMERAS, OmniSceneDataset, sha256_file
from omniscene.config import load_config
from omniscene.geometry import camera_rays
from omniscene.io import write_json


def reference_functions(path, names, namespace):
    """Load only inspected functions, avoiding the references' framework imports."""
    tree = ast.parse(Path(path).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    for n in selected:
        n.returns = None
        for arg in n.args.args:
            arg.annotation = None
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depthsplat', default='/home/dzp62442/Projects/depthsplat')
    parser.add_argument('--svfgs', default='/home/dzp62442/Projects/SVF-GS')
    parser.add_argument('--output', default='work_dirs/smoke/protocol.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = Path(args.depthsplat) / 'src/dataset/utils_omniscene.py'
    reference = reference_functions(source, ['HWC3', 'load_conditions', 'load_info'],
                                    dict(np=np, PIL=PIL, Image=Image, torch=torch, json=json))
    svf_source = Path(args.svfgs) / 'data/transforms/loading.py'
    svf = reference_functions(svf_source, ['load_info'], dict(np=np))
    svf = reference_functions(Path(args.svfgs) / 'model/utils/ops.py', ['get_ray_directions', 'get_rays'],
                              dict(svf, torch=torch, F=torch.nn.functional))
    report = dict(reference_hashes={str(p): sha256_file(p) for p in (source, svf_source)}, checks=[])
    for shape in ('112x200', '224x400'):
        cfg = load_config(f'configs/experiment/omniscene_{shape}.yaml')
        data = OmniSceneDataset(cfg.Dataset, 'mini', 3)
        training_data = OmniSceneDataset(cfg.Dataset, 'val', 2)
        for index in (0, 1):
            sample = data[index]
            token = data.bin_tokens[index]
            with (data.root / cfg.Dataset.version / 'bin_infos_3.2m' / f'{token}.pkl').open('rb') as f:
                sensors = pickle.load(f)['sensor_info']
            infos = [sensors[c][i] for c in CAMERAS for i in (1, 2)] + [sensors[c][0] for c in CAMERAS]
            paths = [str(Path(cfg.Dataset.root) / Path(i['data_path']).relative_to(cfg.Dataset.source_prefix)) for i in infos]
            rgb, masks, k, relative = reference['load_conditions'](paths, list(cfg.Dataset.image_shape), load_rel_depth=True)
            h, w = cfg.Dataset.image_shape
            k[:, 0] *= w
            k[:, 1] *= h
            torch.testing.assert_close(sample['target']['image'], rgb, rtol=0, atol=0)
            torch.testing.assert_close(sample['target']['intrinsics'], k)
            torch.testing.assert_close(sample['target']['rel_depth'], relative, rtol=0, atol=0)
            for view, info in enumerate(infos):
                _, cv_pose, _ = reference['load_info'](info)
                torch.testing.assert_close(sample['target']['extrinsics'][view], torch.tensor(cv_pose).float())
                _, gl_pose, _ = svf['load_info'](info)
                direction = svf['get_ray_directions'](h, w, (k[view, 0, 0], k[view, 1, 1]),
                                                       (k[view, 0, 2], k[view, 1, 2]))
                ro, rd = svf['get_rays'](direction, torch.tensor(gl_pose).float(), keepdim=True, normalize=False)
                ours_o, ours_d = camera_rays(k[view][None, None], sample['target']['extrinsics'][view][None, None], h, w)
                torch.testing.assert_close(ours_o[0, 0], ro)
                torch.testing.assert_close(ours_d[0, 0], rd, rtol=1e-5, atol=1e-6)
            # Use exactly this mini token in the training loader to check floating masks.
            training_data.bin_tokens = [token]
            train = training_data[0]
            torch.testing.assert_close(train['target']['loss_mask'][:12].bool(), masks[:12])
            assert (train['target']['loss_mask'][12:] == 1).all()
            for view, info in enumerate(infos[:12]):
                with Image.open(training_data.path(info, 'mask')) as m:
                    expected = np.array(m.convert('L').resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32) / 255
                torch.testing.assert_close(train['target']['loss_mask'][view], torch.from_numpy(expected), rtol=0, atol=0)
            report['checks'].append(dict(resolution=shape, token=token, exact_rgb=True, exact_relative_depth=True,
                                         intrinsics=True, physical_camera_rays=True, float_loss_masks=True))
            print(f'PASS {shape} {token}', flush=True)
    report['complete'] = True
    write_json(args.output, report)


if __name__ == '__main__':
    main()
