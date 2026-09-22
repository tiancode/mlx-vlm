"""Exercise startup failures and stop scope in a temporary project with fake services."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServiceScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "logs").mkdir()
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"quantization": {
            "group_size": 32, "bits": 8, "mode": "mxfp8"}}))
        (model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"w": "model.safetensors"}}))
        (model / "model.safetensors").touch()
        (model / "model_index.json").write_text('{}')
        for name in ("start.sh", "stop.sh", "start_image.sh"):
            shutil.copy2(ROOT / name, self.root / name)
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
                    "PYTHON": str(self.bin / "python"), "MODEL": str(model), "DRAFT": "",
                    "IMAGE_ENABLED": "0", "PREWARM": "0", "PORT": "32135",
                    "BACKEND_PORT": "32136", "IMAGE_PORT": "32138"}
        self.env.update(IMAGE_PYTHON=self.env['PYTHON'], IMAGE_MODEL=str(model))
        self.script("python", '''
import os, sys
code = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '-c' else ''
if 'with socket.socket() as probe' in code:
    sys.exit(int(os.environ.get('TEST_OCCUPIED', '0')))
if 's.connect_ex' in code:
    sys.exit(0 if os.environ.get('TEST_OCCUPIED') == '1' else 1)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
''')
        self.script("sleep", "import time\ntime.sleep(0.02)\n")
        self.script("pgrep", "import sys\nsys.exit(0)\n")
        self.script("curl", '''
import json, os, sys
if ':32135/' in sys.argv[-1]:
    sys.exit(1)
if sys.argv[-1].endswith('/v1/models'):
    print(json.dumps({'data': [{'id': os.environ['MODEL']}]}))
''')
        (self.root / "service_runner.py").write_text('''
import signal, time
from pathlib import Path
def stop(*args):
    Path('runner-stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path('runner-started').touch()
while True:
    time.sleep(0.01)
''')
        (self.root / "stop_image.sh").write_text("exit 0\n")

    def script(self, name, content):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + content)
        path.chmod(0o755)

    def run_script(self, name):
        return subprocess.run(["bash", str(self.root / name)], env=self.env,
                              capture_output=True, text=True, timeout=15)

    def test_occupied_port_preserves_supervisor_record(self):
        marker = self.root / "logs/start.pid"
        marker.write_text("existing supervisor")
        self.env["TEST_OCCUPIED"] = "1"
        result = self.run_script("start.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "existing supervisor")
        self.assertFalse((self.root / "runner-started").exists())

    def test_proxy_timeout_exits_and_stops_owned_child(self):
        result = self.run_script("start.sh")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("代理就绪检查超时", result.stderr)
        self.assertTrue((self.root / "runner-started").exists())
        self.assertTrue((self.root / "runner-stopped").exists())
        self.assertFalse((self.root / "logs/start.pid").exists())

    def test_occupied_image_port_preserves_runner_record(self):
        marker = self.root / "logs/image-runner.pid"
        marker.write_text("existing runner")
        self.env["TEST_OCCUPIED"] = "1"
        result = self.run_script("start_image.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "existing runner")
        self.assertFalse((self.root / "runner-started").exists())

    def test_stop_scopes_proxy_to_project(self):
        self.script("pgrep", '''
from pathlib import Path
import sys
with Path('patterns').open('a') as f:
    f.write(sys.argv[-1] + '\\n')
sys.exit(1)
''')
        result = self.run_script("stop.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        patterns = (self.root / "patterns").read_text().splitlines()
        self.assertIn(str(self.root), patterns[0])
        self.assertNotEqual(patterns[0], "model_proxy[.]py( |$)")

    def test_stop_reports_occupied_port_as_failure(self):
        self.env["TEST_OCCUPIED"] = "1"
        result = self.run_script("stop.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("全部停止", result.stdout)


if __name__ == "__main__":
    unittest.main()
