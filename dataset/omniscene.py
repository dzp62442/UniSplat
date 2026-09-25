"""OmniScene's existing assets, using the SVF-GS 6 input / 12+6 target protocol.

Only camera calibration is read from the bin metadata. No LiDAR measurements,
sky masks, Metric3D confidence or extra offline assets are accessed.
"""
import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_depth_from_disparity(disp):
    """Use the reference loaders' conversion without judging disparity values."""
    disp = np.asarray(disp, dtype=np.float32)
    # Zero variance may yield NaN. Preserve it as an undefined PCC, without
    # rejecting the bin or replacing its reference depth with fabricated values.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = min(disp.max() / (disp.min() + 0.001), 50.0)
        depth = 1.0 / np.maximum(disp, disp.max() / ratio)
        return (depth - depth.min()) / (depth.max() - depth.min())


def asset_path(data_path, root, source_prefix, role):
    path = Path(data_path)
    if path.is_absolute():
        try:
            path = path.relative_to(source_prefix)
        except ValueError:
            path = path.relative_to(root)
    parts = list(path.parts)
    indices = [i for i, item in enumerate(parts) if item in ('samples', 'sweeps')]
    if len(indices) != 1:
        raise ValueError(f'Expected one samples/sweeps component: {data_path}')
    idx = indices[0]
    suffix = {'rgb': '_small', 'intrinsics': '_param_small', 'metric': '_dptm_small',
              'mask': '_mask_small', 'relative': '_dpt_small'}[role]
    parts[idx] += suffix
    result = Path(root).joinpath(*parts)
    if role == 'metric':
        return result.with_name(result.stem + '_dpt.npy')
    return result.with_suffix({'rgb': '.jpg', 'intrinsics': '.json', 'mask': '.png',
                               'relative': '.npy'}[role])


class OmniSceneDataset(Dataset):
    def __init__(self, cfg, split, stage=3):
        if split not in ('train', 'val', 'mini', 'total') or stage not in (1, 2, 3):
            raise ValueError(f'Invalid split/stage: {split}/{stage}')
        self.cfg, self.split, self.stage = cfg, split, stage
        self.root, self.shape = Path(cfg.root), tuple(cfg.image_shape)
        self.load_metric_depth = split in ('train', 'val') and stage in (1, 2)
        self.load_relative_depth = split in ('mini', 'total')
        self.load_loss_mask = split in ('train', 'val') and stage in (2, 3)
        name = 'train' if split == 'train' else 'val'
        self.manifest = self.root / cfg.version / f'bins_{name}_3.2m.json'
        bins = json.loads(self.manifest.read_text())['bins']
        if not bins:
            raise ValueError(f'No samples to load: {self.manifest}')
        if len(bins) != len(set(bins)):
            logging.getLogger('unisplat.omniscene').warning(
                'Repeated tokens in %s; preserving manifest order and sampling multiplicity', self.manifest)
        self.bin_tokens = bins[:30000:3000][:10] if split == 'val' else bins
        if split == 'mini':
            self.bin_tokens = bins[0::14][:2048]
        self.manifest_sha256 = sha256_file(self.manifest)
        self.tokens_sha256 = hashlib.sha256(json.dumps(self.bin_tokens).encode()).hexdigest()

    def __len__(self):
        return len(self.bin_tokens)

    def path(self, info, role):
        return asset_path(info['data_path'], self.root, self.cfg.source_prefix, role)

    def _view(self, info, novel=False, metric=False):
        h, w = self.shape
        with self.path(info, 'intrinsics').open() as f:
            k = np.array(json.load(f)['camera_intrinsic'], dtype=np.float64)
        with Image.open(self.path(info, 'rgb')) as source:
            source = source.convert('RGB')
            resized = source.size != (w, h)
            if resized:
                scale_w, scale_h = w / source.width, h / source.height
                # Match the reference loader's default PIL RGB resize (bicubic).
                source = source.resize((w, h))
                k = np.array([[k[0, 0] * scale_w, 0, k[0, 2] * scale_w],
                              [0, k[1, 1] * scale_h, k[1, 2] * scale_h], [0, 0, 1]])
            rgb = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 255
        pose = torch.as_tensor(np.array(info['sensor2lidar_transform']), dtype=torch.float32)
        if pose.shape != (4, 4):
            raise ValueError(f'Expected a 4x4 camera pose: {info["data_path"]}')
        view = dict(image=rgb, intrinsics=torch.as_tensor(k, dtype=torch.float32), extrinsics=pose)
        if self.load_loss_mask:
            mask = np.ones(self.shape, dtype=np.float32)
            if novel:
                with Image.open(self.path(info, 'mask')) as source:
                    source = source.convert('L')
                    if source.size != (w, h):
                        source = source.resize((w, h), Image.Resampling.BILINEAR)
                    mask = np.array(source).astype(np.float32) / 255
            if mask.shape != self.shape:
                raise ValueError('Mask and RGB shape mismatch')
            view['loss_mask'] = torch.from_numpy(mask)
        for role, enabled in [('metric', metric), ('relative', self.load_relative_depth)]:
            if not enabled:
                continue
            array = np.load(self.path(info, role), allow_pickle=False).astype(np.float32)
            if array.shape != self.shape:
                array = np.array(Image.fromarray(array).resize((w, h), Image.Resampling.BILINEAR))
            if array.shape != self.shape:
                raise ValueError(f'{role} depth and RGB shape mismatch')
            if role == 'relative':
                array = relative_depth_from_disparity(array)
            view[role] = torch.from_numpy(array.copy())
        return view

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        file = self.root / self.cfg.version / 'bin_infos_3.2m' / f'{token}.pkl'
        with file.open('rb') as stream:
            sensors = pickle.load(stream)['sensor_info']
        if any(len(sensors[cam]) < 3 for cam in CAMERAS):
            raise ValueError(f'Missing center/novel camera entries: {token}')
        input_infos = [sensors[cam][0] for cam in CAMERAS]
        novel_infos = [sensors[cam][i] for cam in CAMERAS for i in (1, 2)]
        inputs = [self._view(info, metric=self.load_metric_depth) for info in input_infos]
        targets = [self._view(info, novel=True) for info in novel_infos] + inputs
        context = {key: torch.stack([v[key] for v in inputs])
                   for key in ('image', 'intrinsics', 'extrinsics')}
        target = {key: torch.stack([v[key] for v in targets])
                  for key in ('image', 'intrinsics', 'extrinsics')}
        if self.load_loss_mask:
            target['loss_mask'] = torch.stack([v['loss_mask'] for v in targets])
        if self.load_relative_depth:
            target['rel_depth'] = torch.stack([v['relative'] for v in targets])
        supervision = ({'input_metric_depth': torch.stack([v['metric'] for v in inputs])}
                       if self.load_metric_depth else {})
        return dict(context=context, target=target, supervision=supervision,
                    meta=dict(bin_token=token, scene_id=token.rsplit('_bin', 1)[0], split=self.split,
                              view_ids=[f'{cam}:{i}' for cam in CAMERAS for i in (1, 2)]
                                       + [f'{cam}:0' for cam in CAMERAS],
                              input_paths=[str(self.path(i, 'rgb')) for i in input_infos],
                              target_paths=[str(self.path(i, 'rgb')) for i in novel_infos + input_infos]))
