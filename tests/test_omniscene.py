"""CPU contract checks: run with python -m unittest discover -s tests -v."""
import json
import csv
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.omniscene import CAMERAS, OmniSceneDataset, relative_depth_from_disparity
from omniscene.config import REPO_ROOT, load_config, due_events, stage_for_step, config_identity
from omniscene.geometry import pad_images, camera_rays, matching_voxel_features
from omniscene.io import isolated_evaluation, write_json
from omniscene.losses import ReconstructionLoss, masked_rgb_mse
from omniscene.metrics import compute_pcc, view_records, summarize_records
from omniscene.model import StaticUniSplat
from omniscene.training import build_optimizer, optimizer_update, ResumeSampler
from omniscene.checkpoint import find_latest_checkpoint, save_checkpoint, load_training_state, restore_checkpoint
from omniscene.notify import FeishuNotifier, send_one

torch.set_num_threads(2)


class ConfigTests(unittest.TestCase):
    def test_configs_and_global_schedule(self):
        for shape, padded in [('112x200', [112, 210]), ('224x400', [224, 406])]:
            cfg = load_config(f'configs/experiment/omniscene_{shape}.yaml')
            self.assertEqual(list(cfg.Model.encoder_image_shape), padded)
            self.assertEqual(sum(cfg.Train.stage_steps), cfg.Train.max_steps)
            events = [(s, due_events(s, cfg)) for s in range(1, 100002)]
            self.assertEqual(sum(v for _, (v, _) in events), 100)
            self.assertEqual([s for s, (_, m) in events if m], [50000, 60000, 70000, 80000, 90000, 100000, 100001])
        self.assertEqual([stage_for_step(s) for s in [0, 44444, 44445, 77778, 77779, 100001]], [1, 1, 2, 2, 3, 3])

    def test_prohibited_overrides_rejected(self):
        for override in ['Dataset.load_lidar=true', 'Model.temporal_enabled=true', 'Train.max_steps=100000',
                         'Dataset.train_batch_size=2', 'Logging.wandb_mode=online', 'Loss.dyn_loss_weight=0.05']:
            with self.assertRaises(ValueError):
                load_config('configs/experiment/omniscene_112x200.yaml', [override])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        cls.cfg.Dataset.root = str(cls.root)
        meta = cls.root / cls.cfg.Dataset.version
        (meta / 'bin_infos_3.2m').mkdir(parents=True)
        cls.tokens = [f'scenefixture_bin{i:03d}' for i in range(30)]
        for split in ('train', 'val'):
            (meta / f'bins_{split}_3.2m.json').write_text(json.dumps({'bins': cls.tokens}))
        sensors = {}
        for ci, cam in enumerate(CAMERAS):
            sensors[cam] = []
            for role in range(3):
                file = f'{cam}_{role}'
                info = dict(data_path=f'/datasets/nuScenes/samples/{file}.jpg',
                            sensor2lidar_transform=np.eye(4, dtype=np.float32))
                info['sensor2lidar_transform'][0, 3] = ci + role * .1
                sensors[cam].append(info)
                for folder in ('samples_small', 'samples_param_small', 'samples_dptm_small',
                               'samples_dpt_small', 'samples_mask_small'):
                    (cls.root / folder).mkdir(exist_ok=True)
                rgb = np.full((224, 400, 3), ci * 30 + role * 7, np.uint8)
                Image.fromarray(rgb).save(cls.root / 'samples_small' / (file + '.jpg'))
                k = [[210, 0, 190], [0, 215, 105], [0, 0, 1]]
                (cls.root / 'samples_param_small' / (file + '.json')).write_text(json.dumps({'camera_intrinsic': k}))
                if role == 0:
                    np.save(cls.root / 'samples_dptm_small' / (file + '_dpt.npy'), np.full((224, 400), ci + 2, np.float32))
                disp = np.linspace(1, 10, 224 * 400, dtype=np.float32).reshape(224, 400)
                np.save(cls.root / 'samples_dpt_small' / (file + '.npy'), disp)
                if role != 0:
                    mask = np.full((224, 400), 255, np.uint8)
                    mask[:, :103] = 0
                    Image.fromarray(mask).save(cls.root / 'samples_mask_small' / (file + '.png'))
        # Deliberately no LIDAR_TOP metadata, confidence arrays, input masks or sky files.
        for token in cls.tokens:
            with (meta / 'bin_infos_3.2m' / (token + '.pkl')).open('wb') as stream:
                pickle.dump({'sensor_info': sensors}, stream)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_view_identity_and_assets(self):
        for hw in ([112, 200], [224, 400]):
            cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.Dataset))
            cfg.image_shape = hw
            data = OmniSceneDataset(cfg, 'train', 2)
            sample = data[0]
            self.assertEqual(tuple(sample['target']['image'].shape), (18, 3, *hw))
            self.assertEqual(tuple(sample['context']['image'].shape), (6, 3, *hw))
            torch.testing.assert_close(sample['target']['image'][12:], sample['context']['image'])
            self.assertEqual(sample['meta']['view_ids'], [f'{cam}:{j}' for cam in CAMERAS for j in (1, 2)]
                             + [f'{cam}:0' for cam in CAMERAS])
            self.assertEqual(float(sample['context']['intrinsics'][0, 0, 0]), 210 * hw[1] / 400)
            self.assertEqual(sample['target']['loss_mask'].dtype, torch.float32)
            self.assertTrue((sample['target']['loss_mask'][12:] == 1).all())
            if hw[0] == 112:
                mask = sample['target']['loss_mask'][:12]
                self.assertTrue(((mask > 0) & (mask < 1)).any())
            self.assertNotIn('rel_depth', sample['target'])
            self.assertEqual(set(sample['context']), {'image', 'intrinsics', 'extrinsics'})

    def test_stage3_and_test_never_read_metric_or_mask(self):
        stage3 = OmniSceneDataset(self.cfg.Dataset, 'train', 3)
        self.assertEqual(stage3[0]['supervision'], {})
        for split in ('mini', 'total'):
            data = OmniSceneDataset(self.cfg.Dataset, split, 3)
            original = data.path
            def restricted(info, role):
                self.assertNotIn(role, ('metric', 'mask'))
                return original(info, role)
            data.path = restricted
            sample = data[0]
            self.assertEqual(sample['supervision'], {})
            self.assertNotIn('loss_mask', sample['target'])
            self.assertEqual(tuple(sample['target']['rel_depth'].shape), (18, 112, 200))
        self.assertEqual(len(OmniSceneDataset(self.cfg.Dataset, 'mini')), 3)
        self.assertEqual(len(OmniSceneDataset(self.cfg.Dataset, 'total')), 30)

    def test_degenerate_reference_is_explicit(self):
        self.assertTrue(np.isnan(relative_depth_from_disparity(np.ones((2, 2)))).all())
        disparity = np.array([[1., 2.], [3., 4.]], np.float32)
        expected = 1 / np.maximum(disparity, disparity.min() + .001)
        expected = (expected - expected.min()) / (expected.max() - expected.min())
        np.testing.assert_allclose(relative_depth_from_disparity(disparity), expected)

    def test_negative_disparity_uses_reference_conversion(self):
        disparity = np.array([[-.5, 1.], [2., 3.]], np.float32)
        ratio = min(disparity.max() / (disparity.min() + .001), 50.)
        expected = 1. / np.maximum(disparity, disparity.max() / ratio)
        expected = (expected - expected.min()) / (expected.max() - expected.min())
        np.testing.assert_allclose(relative_depth_from_disparity(disparity), expected)

    def test_assets_resize_independently_of_rgb(self):
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.Dataset))
        cfg.image_shape = [224, 400]  # RGB already has the target dimensions.
        mask_path = self.root / 'samples_mask_small' / f'{CAMERAS[0]}_1.png'
        original = mask_path.read_bytes()
        try:
            Image.fromarray(np.full((112, 200), 128, np.uint8)).save(mask_path)
            with patch('dataset.omniscene.np.load', return_value=np.ones((112, 200), np.float32)):
                sample = OmniSceneDataset(cfg, 'train', 2)[0]
                self.assertEqual(sample['supervision']['input_metric_depth'].shape, (6, 224, 400))
                self.assertEqual(sample['target']['loss_mask'].shape, (18, 224, 400))
                self.assertEqual(sample['context']['intrinsics'][0, 0, 0], 210)
        finally:
            mask_path.write_bytes(original)

    def test_repeated_manifest_entries_preserve_sampling(self):
        manifest = self.root / self.cfg.Dataset.version / 'bins_val_3.2m.json'
        original = manifest.read_bytes()
        try:
            manifest.write_text(json.dumps({'bins': self.tokens + self.tokens[:1]}))
            with self.assertLogs('unisplat.omniscene', level='WARNING'):
                data = OmniSceneDataset(self.cfg.Dataset, 'total')
            self.assertEqual(data.bin_tokens, self.tokens + self.tokens[:1])
        finally:
            manifest.write_bytes(original)


class GeometryTests(unittest.TestCase):
    def test_padding_preserves_pixels_and_rays(self):
        for h, w in ((112, 200), (224, 400)):
            images = torch.rand(1, 6, 3, h, w)
            padded = pad_images(images)
            self.assertEqual(padded.shape[-1] % 14, 0)
            torch.testing.assert_close(padded[..., :w], images)
            k = torch.tensor([[200., 0, w * .46], [0, 195, h * .43], [0, 0, 1]])[None, None].repeat(1, 6, 1, 1)
            pose = torch.eye(4)[None, None].repeat(1, 6, 1, 1)
            a, b = camera_rays(k, pose, h, w)
            c, d = camera_rays(k, pose, h, padded.shape[-1])
            torch.testing.assert_close(a, c[:, :, :, :w])
            torch.testing.assert_close(b, d[:, :, :, :w])
            # SVF-GS flips pose Y/Z and also flips ray Y/Z: same reference-space ray.
            flip = torch.diag(torch.tensor([1., -1., -1.]))
            gl_rays = torch.einsum('ij,bvhwj->bvhwi', flip, b)
            torch.testing.assert_close(torch.einsum('ij,bvhwj->bvhwi', flip, gl_rays), b)
            self.assertTrue((b[..., 2] == 1).all())

    def test_exact_voxel_lookup_and_gradient(self):
        feats = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        centers = torch.tensor([[.1, .1, .2], [.5, .1, .2]])
        points = torch.tensor([[.01, .02, .03], [.45, .05, .1], [.25, .05, .1], [-.1, .1, .1]])
        out = matching_voxel_features(points, feats, centers, torch.zeros(3), torch.tensor([.1, .1, .2]),
                                      torch.tensor([10, 10, 10]))
        torch.testing.assert_close(out, torch.tensor([[1., 2.], [3., 4.], [0., 0.], [0., 0.]]))
        out.sum().backward()
        torch.testing.assert_close(feats.grad, torch.ones_like(feats))


class MetricTests(unittest.TestCase):
    def test_floating_mask_is_squared_without_renormalizing(self):
        pred = torch.ones(2, 3, 2, 2)
        gt = torch.zeros_like(pred)
        mask = torch.tensor([[[0., .5], [1., 1.]], [[1., 1.], [1., 1.]]])
        self.assertAlmostEqual(masked_rgb_mse(pred, gt, mask).item(), mask.square().mean().item())

    def test_pcc_grouping_is_not_average_per_view(self):
        reference = torch.arange(18 * 12, dtype=torch.float32).reshape(18, 3, 4)
        predicted = reference + torch.arange(18).reshape(18, 1, 1) * -20
        scores = {k: torch.arange(18, dtype=torch.float32) for k in ('psnr', 'ssim', 'lpips')}
        rows = view_records('bin', 'scene', scores, reference, predicted, (3, 4))
        self.assertAlmostEqual(rows[0]['pcc'], compute_pcc(reference, predicted).item())
        self.assertNotAlmostEqual(rows[0]['pcc'], 1.)
        self.assertAlmostEqual(rows[1]['pcc'], compute_pcc(reference[:12], predicted[:12]).item())
        self.assertTrue(torch.isnan(compute_pcc(torch.ones(4), torch.arange(4.))))
        self.assertTrue(summarize_records(rows, ['bin'])['complete'])
        self.assertFalse(summarize_records(rows + rows, ['bin'])['complete'])
        self.assertFalse(summarize_records(rows, ['bin', 'missing'])['complete'])
        rows[0]['pcc'] = float('nan')
        summary = summarize_records(rows, ['bin'])
        self.assertTrue(summary['complete'])
        self.assertFalse(summary['metrics_finite'])
        self.assertIsNone(summary['all_18']['pcc'])
        self.assertEqual(summary['all_18']['finite_metric_counts']['pcc'], 0)
        self.assertEqual(summary['novel_12']['finite_metric_counts']['pcc'], 1)
        self.assertEqual(summary['all_18']['psnr'], 8.5)

    def test_repeated_expected_tokens_count_as_manifest_entries(self):
        scores = {k: torch.ones(18) for k in ('psnr', 'ssim', 'lpips')}
        depth = torch.arange(18 * 12, dtype=torch.float32).reshape(18, 3, 4)
        rows = view_records('bin', 'scene', scores, depth, depth, (3, 4))
        summary = summarize_records(rows + rows, ['bin', 'bin'])
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['expected_bins'], 2)
        self.assertEqual(summary['all_18']['num_records'], 2)
        self.assertFalse(summarize_records(rows, ['bin', 'bin'])['complete'])

    def test_undefined_pcc_has_reason_without_losing_rgb_scores(self):
        scores = {k: torch.ones(18) for k in ('psnr', 'ssim', 'lpips')}
        depth = torch.arange(18 * 12, dtype=torch.float32).reshape(18, 3, 4)
        for reference, reason in [(torch.ones_like(depth), 'zero_variance_or_insufficient_pixels'),
                                  (torch.full_like(depth, float('nan')), 'nonfinite_reference')]:
            rows = view_records('bin', 'scene', scores, reference, depth, (3, 4))
            self.assertEqual(rows[0]['pcc_status'], reason)
            summary = summarize_records(rows, ['bin'])
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['all_18']['ssim'], 1.)
            self.assertEqual(summary['all_18']['nonfinite_metrics'][0]['reason'], reason)


class TinyModel(StaticUniSplat):
    """Exercise production freeze/optimizer/checkpoint code without allocating Pi3."""
    def __init__(self):
        nn.Module.__init__(self)
        self.geometry_model = nn.Linear(2, 2)
        self.gaussian_head = nn.Module()
        for name in ('point_decoder', 'scale_head', 'shift_head', 'image_backbone', 'unet', 'to_gaussians_sky'):
            setattr(self.gaussian_head, name, nn.Linear(2, 2))
        self.gaussian_head.perceptual_loss = nn.Identity()
        self.stage = 1
        self.set_stage(1)


class CheckpointDiscoveryTests(unittest.TestCase):
    def checkpoint(self, run, step):
        path = Path(run) / 'checkpoints' / f'step_{step:06d}'
        path.mkdir(parents=True)
        for name in ('manifest.json', 'model.safetensors', 'training_state.pth'):
            (path / name).write_text('fixture')
        return path

    def test_new_run_and_interrupted_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(find_latest_checkpoint(d))
            (Path(d) / 'checkpoints/step_002000.partial.tmp').mkdir(parents=True)
            self.assertIsNone(find_latest_checkpoint(d))
            saved = self.checkpoint(d, 1000)
            self.assertEqual(find_latest_checkpoint(d), saved)

    def test_latest_link_may_be_missing_stale_or_broken(self):
        with tempfile.TemporaryDirectory() as d:
            older = self.checkpoint(d, 1000)
            newest = self.checkpoint(d, 2000)
            root = Path(d) / 'checkpoints'
            self.assertEqual(find_latest_checkpoint(d), newest)
            link = root / 'latest'
            link.symlink_to(older.name)
            self.assertEqual(find_latest_checkpoint(d), newest)
            link.unlink()
            link.symlink_to('step_003000')
            self.assertEqual(find_latest_checkpoint(d), newest)

    def test_damaged_committed_checkpoint_never_restarts_silently(self):
        with tempfile.TemporaryDirectory() as d:
            self.checkpoint(d, 1000)
            latest = self.checkpoint(d, 2000)
            (latest / 'training_state.pth').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Newest checkpoint is incomplete'):
                find_latest_checkpoint(d)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'checkpoints'
            root.mkdir()
            (root / 'latest').symlink_to('step_001000')
            with self.assertRaisesRegex(FileNotFoundError, 'No committed'):
                find_latest_checkpoint(d)

    def test_resolutions_and_custom_work_directories_are_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            small = Path(d) / 'unisplat_omniscene_static_112x200'
            large = Path(d) / 'unisplat_omniscene_static_224x400'
            custom = Path(d) / 'custom'
            small_ckpt = self.checkpoint(small, 1000)
            self.checkpoint(large, 90000)
            self.assertEqual(find_latest_checkpoint(small), small_ckpt)
            self.assertIsNone(find_latest_checkpoint(custom))


class TrainingTests(unittest.TestCase):
    def test_empty_alignment_and_loss_do_not_raise(self):
        from model.gaussian_head.static_head import StaticGaussianHead
        cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        depth = torch.ones(1, 6, 14, 14)
        intrinsics = torch.tensor([[10., 0, 7], [0, 10., 7], [0, 0, 1]])[None, None].repeat(1, 6, 1, 1)
        head = SimpleNamespace(cfg=cfg.Model.Gaussian_head)
        for unavailable in (torch.zeros_like(depth), torch.full_like(depth, float('nan'))):
            alignment = StaticGaussianHead.align_depth(head, depth, unavailable, intrinsics)
            self.assertFalse(alignment[2].any())
            rec = dict(alignment=alignment, pred_scale=torch.ones(1, 6, requires_grad=True),
                       pred_shift=torch.ones(1, 6, requires_grad=True))
            loss = sum(ReconstructionLoss(cfg, nn.Identity())(None, rec, {}, 1).values())
            self.assertEqual(loss.item(), 0)
            loss.backward()
            torch.testing.assert_close(rec['pred_scale'].grad, torch.zeros(1, 6))

    def test_numerical_skips_leave_optimizer_intact_and_allow_next_update(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(2.))
                self.mode = 'healthy'
                self.p.register_hook(lambda g: g * float('nan') if self.mode == 'gradient' else g)
            def forward(self, context, stage, supervision):
                if self.mode == 'geometry':
                    raise FloatingPointError('nonfinite model geometry')
                if self.mode == 'interface':
                    raise ValueError('fixture programming error')
                result = dict(loss=self.p.square())
                if self.mode == 'loss':
                    result['loss'] *= float('nan')
                if self.mode == 'supervision':
                    result['alignment'] = (None, None, torch.zeros(1, 6, dtype=torch.bool))
                return result
        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cpu')
        criterion = lambda m, r, t, s: {'rec': r['loss']}
        batch = dict(context={}, target={}, supervision={})
        for mode in ('supervision', 'loss', 'gradient', 'geometry'):
            model.mode = mode
            for _ in range(12):
                _, _, reason = optimizer_update(model, criterion, optimizer, scaler, batch, 1, 10.)
                self.assertIsNotNone(reason)
                self.assertEqual(model.p.item(), 2.)
                self.assertEqual(len(optimizer.state), 0)
        model.mode = 'healthy'
        _, _, reason = optimizer_update(model, criterion, optimizer, scaler, batch, 1, 10.)
        self.assertIsNone(reason)
        self.assertEqual(optimizer.state[model.p]['step'], 1)
        self.assertLess(model.p.item(), 2.)
        model.mode = 'interface'
        with self.assertRaisesRegex(ValueError, 'programming error'):
            optimizer_update(model, criterion, optimizer, scaler, batch, 1, 10.)

    def test_stage_transition_reenables_parameters(self):
        cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        model = TinyModel()
        for stage in (1, 2, 3, 1):
            optimizer, _ = build_optimizer(model, cfg, stage)
            for name, p in model.named_parameters():
                scale = any(name.startswith('gaussian_head.' + n + '.') for n in ('point_decoder', 'scale_head', 'shift_head'))
                expected = scale if stage == 1 else name.startswith('gaussian_head.') and not scale and 'sky' not in name
                self.assertEqual(p.requires_grad, expected, name)
            self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},
                             {id(p) for group in optimizer.param_groups for p in group['params']})

    def test_sampler_prefetch_does_not_advance_resume(self):
        full = list(ResumeSampler(40, 42, 1, 0))
        sampler = ResumeSampler(40, 42, 1, 8)
        _ = list(sampler)
        self.assertEqual(list(sampler), full[8:])

    def test_evaluation_restores_rng_and_mode(self):
        model = TinyModel().train()
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        with isolated_evaluation(model):
            torch.rand(50)
            self.assertFalse(model.training)
        torch.testing.assert_close(torch.rand(3), expected)
        self.assertTrue(model.training)
        self.assertFalse(model.geometry_model.training)

    def test_checkpoint_optimizer_rng_pairing(self):
        cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        model = TinyModel()
        optimizer, scheduler = build_optimizer(model, cfg, 1)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step()
        scheduler.step()
        progress = dict(global_step=1, stage=1, stage_step=1, epoch=0, offset=1,
                        val_count=0, last_validation_step=0, last_mini_test_step=0)
        with tempfile.TemporaryDirectory() as d:
            path = save_checkpoint(d, model, optimizer, scheduler, scaler, progress, cfg, {'train': 'hash'})
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            expected_rng = torch.rand(5)
            new = TinyModel()
            opt2, sch2 = build_optimizer(new, cfg, 1)
            state, manifest = load_training_state(path, cfg, {'train': 'hash'})
            restore_checkpoint(path, new, opt2, sch2, scaler, state)
            for k, v in expected.items():
                torch.testing.assert_close(new.state_dict()[k], v)
            torch.testing.assert_close(torch.rand(5), expected_rng)
            self.assertEqual(scheduler.last_epoch, sch2.last_epoch)
            self.assertEqual(manifest['progress'], progress)
            with self.assertRaises(ValueError):
                load_training_state(path, cfg, {'train': 'different'})
            with self.assertRaisesRegex(ValueError, 'configuration or data'):
                load_training_state(path, load_config('configs/experiment/omniscene_224x400.yaml'), {'train': 'hash'})
            with (path / 'training_state.pth').open('ab') as f:
                f.write(b'damaged')
            with self.assertRaisesRegex(ValueError, 'training state hash mismatch'):
                load_training_state(path, cfg, {'train': 'hash'})

    def test_final_evaluation_resume_does_not_repeat_training(self):
        """Run the real loop with tiny CPU stand-ins and interrupt the final mini."""
        self.exercise_auto_resume()

    def test_training_and_resume_accept_undefined_pcc(self):
        self.exercise_auto_resume(undefined_pcc=True)

    def test_training_auto_resume_across_stage_boundaries(self):
        for step in (2, 3, 5):
            with self.subTest(interrupted_checkpoint=step):
                self.exercise_auto_resume(interrupted_checkpoint=step)

    def exercise_auto_resume(self, interrupted_checkpoint=None, undefined_pcc=False):
        from omniscene import training
        cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        cfg.Train.stage_steps, cfg.Train.max_steps = [2, 3, 2], 7
        cfg.Train.val_every_steps, cfg.Train.mini_test_every_n_val = 1, 2
        cfg.Dataset.num_workers = 0
        cfg.Feishu.enabled = False
        updates, validations, evaluations = [], [], []

        class Data(torch.utils.data.Dataset):
            manifest_sha256, tokens_sha256 = 'manifest', 'tokens'
            def __init__(self, config, split, stage=1):
                self.bin_tokens = ['0'] if split == 'mini' else [str(i) for i in range(20)]
            def __len__(self):
                return len(self.bin_tokens)
            def __getitem__(self, i):
                return dict(context={'image': torch.tensor(float(i))}, target={}, supervision={},
                            meta={'bin_token': str(i)})

        class Model(TinyModel):
            def cuda(self):
                return self
            def forward(self, context, stage, supervision):
                updates.append((stage, context['image'].item()))
                return {'loss': sum(p.square().sum() for p in self.parameters() if p.requires_grad)}

        class Loss(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def cuda(self):
                return self
            def forward(self, model, rec, target, stage):
                return {'rec': rec['loss']}

        def validation(model, *args):
            validations.append(model.stage)
            return {'stage': model.stage}

        def evaluation(model, config, output, identity, **kwargs):
            step = identity['global_step']
            evaluations.append(step)
            if interrupted_checkpoint is None and step == 7 and evaluations.count(7) == 1:
                raise RuntimeError('injected final evaluation interruption')
            rows = [dict(bin_token='0', scene_id='scene', view_group=g, height=112, width=200,
                         psnr=20., ssim=.5, lpips=.4, pcc=float('nan') if undefined_pcc else .3,
                         global_step=step, checkpoint_sha256=identity['sha256'])
                    for g in ('all_18', 'novel_12', 'input_6')]
            result = dict(summarize_records(rows, ['0']), timing={})
            write_json(Path(output) / 'data_provenance.json',
                       training.evaluation_identity(config, Data(config.Dataset, 'mini'), identity, 'mini'))
            write_json(Path(output) / 'model_parameters.json', {})
            write_json(Path(output) / 'reconstruction_time.json', {'num_bins': 1, 'per_bin': [{'bin_token': '0'}]})
            with (Path(output) / 'per_bin_metrics.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(Path(output) / 'evaluation_summary.json', result)
            return result

        real_scaler = torch.amp.GradScaler
        def save_and_interrupt(*args, **kwargs):
            path = save_checkpoint(*args, **kwargs)
            if int(path.name[5:]) == interrupted_checkpoint:
                raise RuntimeError('injected training interruption after checkpoint commit')
            return path

        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            weights = Path(d) / 'weights'
            weights.write_text('fixture')
            stack.enter_context(patch.object(training, 'require_cuda', lambda: None))
            stack.enter_context(patch.object(torch.cuda, 'set_device', lambda *args: None))
            stack.enter_context(patch.object(training, 'StaticUniSplat', lambda *a, **kw: Model()))
            stack.enter_context(patch.object(training, 'OmniSceneDataset', Data))
            stack.enter_context(patch.object(training, 'ReconstructionLoss', Loss))
            stack.enter_context(patch.object(training, 'validate', validation))
            stack.enter_context(patch.object(training, 'evaluate', evaluation))
            stack.enter_context(patch.object(training, 'save_checkpoint', save_and_interrupt))
            stack.enter_context(patch.object(training, 'local_path', lambda p: weights))
            stack.enter_context(patch('omniscene.evaluate.local_path', lambda p: weights))
            stack.enter_context(patch.object(training, 'to_device', lambda tree, device: tree))
            stack.enter_context(patch.object(torch.amp, 'GradScaler', lambda *a: real_scaler('cuda', enabled=False)))
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                training.run_training(cfg, work_dir=d)
            self.assertEqual(len(updates), interrupted_checkpoint or 7)
            self.assertFalse((Path(d) / 'training_complete.json').exists())
            (Path(d) / 'checkpoints/latest').unlink()
            training.run_training(cfg, work_dir=d)
            self.assertEqual(len(updates), 7)
            self.assertEqual([s for s, _ in updates], [1, 1, 2, 2, 2, 3, 3])
            self.assertEqual([i for _, i in updates], list(ResumeSampler(20, 42, 0, 0))[:7])
            self.assertEqual(len(validations), 7)
            expected_evaluations = [4, 6, 7, 7] if interrupted_checkpoint is None else [4, 6, 7]
            self.assertEqual(evaluations, expected_evaluations)
            progress = json.loads((Path(d) / 'training_complete.json').read_text())
            self.assertEqual(progress['last_mini_test_step'], 7)
            self.assertEqual(progress['offset'], 7)
            # Exact model parity with an uninterrupted reference optimizer sequence.
            torch.manual_seed(42)
            reference = TinyModel()
            for stage, steps in enumerate(cfg.Train.stage_steps, 1):
                opt, schedule = build_optimizer(reference, cfg, stage)
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    sum(p.square().sum() for p in reference.parameters() if p.requires_grad).backward()
                    torch.nn.utils.clip_grad_norm_((p for p in reference.parameters() if p.requires_grad), cfg.Train.grad_max_norm)
                    opt.step()
                    schedule.step()
            from safetensors.torch import load_file
            actual = load_file(str(Path(d) / 'checkpoints/step_000007/model.safetensors'))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
            # A completed run exits before model allocation, CUDA access or new notifications.
            with patch.object(training, 'StaticUniSplat', side_effect=AssertionError('must not build model')), \
                 patch.object(training, 'require_cuda', side_effect=AssertionError('must not need CUDA')), \
                 patch.object(training, 'FeishuNotifier', side_effect=AssertionError('must not notify')):
                training.run_training(cfg, work_dir=d)
                with patch.object(training, 'find_latest_checkpoint', side_effect=AssertionError('explicit resume wins')):
                    training.run_training(cfg, resume=Path(d) / 'checkpoints/step_000007', work_dir=d)
            self.assertEqual(evaluations, expected_evaluations)
            # A stale completion marker cannot hide a missing final evaluation artifact.
            (Path(d) / 'evaluation/step_000007/mini/per_bin_metrics.csv').unlink()
            training.run_training(cfg, work_dir=d)
            self.assertEqual(len(updates), 7)
            self.assertEqual(evaluations, expected_evaluations + [7])


class NotificationTests(unittest.TestCase):
    def test_outbox_is_bounded_deduplicated_and_retried(self):
        cfg = load_config('configs/experiment/omniscene_112x200.yaml').Feishu
        cfg.max_attempts = 2
        with tempfile.TemporaryDirectory() as d:
            module = Path(d) / 'auto_monitor/send_feishu'
            module.mkdir(parents=True)
            (module / '__init__.py').write_text('')
            cfg.module_root = d
            with patch('omniscene.notify.subprocess.run', return_value=SimpleNamespace(returncode=1)) as run:
                notifier = FeishuNotifier(cfg, d)
                notifier.emit('train_start', 0, 'fixture', 'no real notification')
                notifier.emit('train_start', 0, 'duplicate', 'must be ignored')
                notifier.close()
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args.kwargs['timeout'], cfg.timeout_seconds)
            path = Path(d) / 'notifications/train_start_000000.json'
            self.assertEqual(json.loads(path.read_text())['status'], 'pending')
            with patch('omniscene.notify.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
                notifier = FeishuNotifier(cfg, d)
                notifier.close()
                self.assertEqual(run.call_count, 1)
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'sent')
            self.assertEqual(result['attempts'], 3)
            with patch('omniscene.notify.importlib.import_module',
                       return_value=SimpleNamespace(send_feishu=lambda *a: False)):
                with self.assertRaises(RuntimeError):
                    send_one(path)


class EvaluationCacheTests(unittest.TestCase):
    def test_numerical_failures_and_undefined_pcc_continue_and_cache(self):
        from omniscene import evaluate as module
        cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        cfg.Dataset.num_workers, cfg.Evaluation.warmup_steps = 0, 2
        depth = torch.arange(18 * 12, dtype=torch.float32).reshape(18, 3, 4)
        class Data(torch.utils.data.Dataset):
            manifest_sha256, tokens_sha256, bin_tokens = 'a', 'b', ['0', '1', '2']
            def __len__(self):
                return 3
            def __getitem__(self, i):
                return dict(context={'image': torch.tensor(i)},
                            target=dict(image=torch.zeros(18, 3, 3, 4), intrinsics=torch.zeros(18, 3, 3),
                                        extrinsics=torch.zeros(18, 4, 4),
                                        rel_depth=torch.ones_like(depth) if i == 1 else depth),
                            meta=dict(bin_token=str(i), scene_id='fixture'))
        class Model(TinyModel):
            def __init__(self):
                super().__init__()
                self.gaussian_head.rope = nn.Identity()
                self.calls = []
            def reconstruct(self, context, stage=3):
                index = context['image'].item()
                self.calls.append(index)
                if index == 0:
                    raise FloatingPointError('fixture geometry failure')
                return {'gaussians': torch.zeros(1, 14)}
            def render(self, gaussians, cameras, shape):
                return dict(image=torch.zeros(18, 3, 3, 4), depth=depth[:, None])
        model = Model()
        identity = dict(path='fixture', sha256='sha', global_step=50)
        metrics = lambda a, b: {k: torch.ones(18) for k in ('psnr', 'ssim', 'lpips')}
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            stack.enter_context(patch.object(module, 'OmniSceneDataset', return_value=Data()))
            stack.enter_context(patch.object(module, 'sha256_file', return_value='weight_sha'))
            stack.enter_context(patch.object(module, 'local_path', return_value=Path(d)))
            stack.enter_context(patch.object(torch.cuda, 'synchronize', lambda *a: None))
            stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='CPU fixture'))
            result = module.evaluate(model, cfg, d, identity, metrics=metrics)
            self.assertTrue(result['complete'])
            self.assertFalse(result['metrics_finite'])
            self.assertEqual(result['all_18']['num_records'], 3)
            self.assertEqual(result['all_18']['finite_metric_counts'], dict(psnr=2, ssim=2, lpips=2, pcc=1))
            self.assertEqual(result['timing']['reconstruction_ms_num_bins'], 2)
            self.assertEqual(result['numerical_failures'][0]['bin_token'], '0')
            self.assertEqual(model.calls, [0, 1, 1, 1, 2])
            with patch.object(model, 'reconstruct', side_effect=AssertionError('must reuse recorded evaluation')):
                self.assertEqual(module.evaluate(model, cfg, d, identity), result)

    def test_validation_continues_after_unavailable_losses(self):
        from omniscene import evaluate as module
        cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        cfg.Dataset.num_workers = 0
        class Data(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, i):
                return dict(context={'image': torch.tensor(i)}, target={}, supervision={}, meta={'bin_token': str(i)})
        class Model(TinyModel):
            def forward(self, context, stage, supervision):
                index = context['image'].item()
                if index == 0:
                    return dict(alignment=(None, None, torch.zeros(1, 6, dtype=torch.bool)))
                if index == 3:
                    raise FloatingPointError('fixture model failure')
                return dict(loss=torch.tensor(float('nan') if index == 1 else 2.))
        with patch.object(module, 'OmniSceneDataset', return_value=Data()):
            result = module.validate(Model(), cfg, lambda m, r, t, s: {'rec': r['loss']})
        self.assertEqual(result['num_bins'], 4)
        self.assertEqual(result['used_bins'], 1)
        self.assertEqual(result['losses'], {'rec': 2.})
        self.assertEqual([r['bin_token'] for r in result['skipped_bins']], ['0', '1', '3'])

    def test_only_complete_matching_artifacts_are_reused(self):
        from omniscene import evaluate as module
        cfg = load_config('configs/experiment/omniscene_112x200.yaml')
        ds = SimpleNamespace(manifest_sha256='a', tokens_sha256='b', bin_tokens=['fixture'])
        class Dataset:
            manifest_sha256, tokens_sha256, bin_tokens = ds.manifest_sha256, ds.tokens_sha256, ds.bin_tokens
            def __len__(self):
                return 1
        identity = dict(path='fixture', sha256='sha', global_step=50)
        rows = [dict(bin_token='fixture', scene_id='scene', view_group=g, height=112, width=200,
                     psnr=20., ssim=.5, lpips=.4, pcc=.3, global_step=50, checkpoint_sha256='sha')
                for g in ('all_18', 'novel_12', 'input_6')]
        expected = summarize_records(rows, ['fixture'])
        with tempfile.TemporaryDirectory() as d, patch.object(module, 'OmniSceneDataset', return_value=Dataset()):
            directory = Path(d)
            write_json(directory / 'evaluation_summary.json', expected)
            write_json(directory / 'model_parameters.json', {})
            write_json(directory / 'reconstruction_time.json', {'num_bins': 1, 'per_bin': [{'bin_token': 'fixture'}]})
            def write_rows(values):
                with (directory / 'per_bin_metrics.csv').open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(values)
            write_rows(rows)
            with patch.object(module, 'sha256_file', return_value='weight_sha'), \
                 patch.object(module, 'local_path', return_value=directory), \
                 patch.object(module, 'ImageMetrics', side_effect=RuntimeError('must recompute')):
                write_json(directory / 'data_provenance.json', module.evaluation_identity(cfg, Dataset(), identity, 'mini'))
                self.assertEqual(module.evaluate(TinyModel(), cfg, directory, identity), expected)
                write_rows(rows[:2])
                with self.assertRaisesRegex(RuntimeError, 'must recompute'):
                    module.evaluate(TinyModel(), cfg, directory, identity)
                with self.assertRaisesRegex(ValueError, 'mix evaluation identities'):
                    module.evaluate(TinyModel(), cfg, directory, dict(identity, sha256='different'))


if __name__ == '__main__':
    unittest.main()
