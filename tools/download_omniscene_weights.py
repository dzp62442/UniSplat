"""Explicit, one-time preparation of official weights; never called by training."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PI3_REVISION = 'ae722e7039287d0c8fde9f11f197f804f44b510c'


def prepare():
    revision = PI3_REVISION
    entries = [
        ('pi3/model.safetensors', f'https://huggingface.co/yyfz233/Pi3/resolve/{revision}/model.safetensors',
         '33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d', None),
        ('dinov2/dinov2_vits14_reg4_pretrain.pth',
         'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth',
         'f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb', None),
        ('lpips/vgg16-397923af.pth', 'https://download.pytorch.org/models/vgg16-397923af.pth',
         '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
         Path.home() / '.cache/torch/hub/checkpoints/vgg16-397923af.pth'),
        ('lpips/vgg.pth', 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1',
         'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868',
         Path.home() / '.cache/torch/hub/checkpoints/vgg.pth'),
    ]

    def download(entry):
        relative, url, expected, cache = entry
        dest = ROOT / 'ckpt' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            part = dest.with_suffix(dest.suffix + '.part')
            if cache is not None and cache.is_file():
                shutil.copyfile(cache, part)
            else:
                subprocess.run(['curl', '--http1.1', '--fail', '--location', '--retry', '5',
                                '--retry-all-errors', '--continue-at', '-', '--connect-timeout', '30',
                                '--speed-time', '120', '--speed-limit', '1024', '--silent', '--show-error',
                                '--output', str(part), url], check=True)
            candidate = part
        else:
            candidate = dest
        h = hashlib.sha256()
        with candidate.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        digest = h.hexdigest()
        if expected and digest != expected:
            raise ValueError(f'Official hash mismatch: {candidate}; remove the invalid file before retrying')
        if candidate != dest:
            candidate.replace(dest)
        print(f'Ready: {relative} ({dest.stat().st_size} bytes, sha256={digest})', flush=True)
        return dict(path=str(dest), url=url, sha256=digest, expected_sha256=expected,
                    bytes=dest.stat().st_size, source_cache=str(cache) if cache and cache.is_file() else None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, entries))
    (ROOT / 'ckpt' / 'sources.json').write_text(json.dumps(
        dict(pi3_revision=revision, prepared_at=time.time(), weights=results), indent=2) + '\n')


if __name__ == '__main__':
    prepare()
