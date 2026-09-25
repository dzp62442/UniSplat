"""Nonblocking send_feishu delivery with a durable outbox and bounded subprocesses."""
import argparse
import importlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from .config import REPO_ROOT
from .io import write_json


class FeishuNotifier:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.directory = Path(output_dir) / 'notifications'
        self.jobs = queue.Queue()
        self.thread = None
        if not cfg.enabled:
            return
        module = Path(cfg.module_root).expanduser() / 'auto_monitor' / 'send_feishu'
        if not (module / '__init__.py').is_file() and not module.with_suffix('.py').is_file():
            raise FileNotFoundError(f'send_feishu module missing: {module}')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._worker, name='feishu-outbox', daemon=True)
        self.thread.start()
        for path in sorted(self.directory.glob('*.json')):
            if json.loads(path.read_text()).get('status') != 'sent':
                self.jobs.put(path)

    def emit(self, event, step, subject, content):
        if not self.cfg.enabled or event not in self.cfg.events:
            return
        path = self.directory / f'{event}_{step:06d}.json'
        if path.exists():
            return
        write_json(path, dict(event=event, step=step, subject=subject, content=content,
                             module_root=str(self.cfg.module_root), status='pending', attempts=0))
        self.jobs.put(path)

    def _worker(self):
        while True:
            path = self.jobs.get()
            try:
                if path is None:
                    return
                payload = json.loads(path.read_text())
                if payload.get('status') == 'sent':
                    continue
                for attempt in range(self.cfg.max_attempts):
                    payload['attempts'] += 1
                    try:
                        result = subprocess.run([sys.executable, '-m', 'omniscene.notify', '--send-one', str(path)],
                                                cwd=REPO_ROOT, capture_output=True,
                                                timeout=self.cfg.timeout_seconds, check=False)
                        if result.returncode:
                            raise RuntimeError(f'helper_exit_{result.returncode}')
                        payload['status'], payload['sent_at'] = 'sent', time.time()
                    except (subprocess.TimeoutExpired, RuntimeError, OSError) as error:
                        payload['status'], payload['last_error'] = 'pending', type(error).__name__
                    write_json(path, payload)
                    if payload['status'] == 'sent':
                        break
            finally:
                self.jobs.task_done()

    def close(self):
        if self.thread is not None:
            self.jobs.put(None)
            self.thread.join(timeout=self.cfg.timeout_seconds * self.cfg.max_attempts + 5)


def send_one(path):
    payload = json.loads(Path(path).read_text())
    sys.path.insert(0, str(Path(payload['module_root']).expanduser()))
    helper = importlib.import_module('auto_monitor.send_feishu')
    result = helper.send_feishu(payload['subject'], payload['content'])
    # Different local helper versions either raise, return False, or return an API result.
    if result is False or (isinstance(result, dict) and result.get('code', 0) != 0):
        raise RuntimeError('send_feishu reported failure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--send-one', required=True)
    send_one(parser.parse_args().send_one)
