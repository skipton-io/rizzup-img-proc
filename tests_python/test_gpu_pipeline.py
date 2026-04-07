import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gpu_pipeline.py"


class GpuPipelineTests(unittest.TestCase):
    def run_script(self, payload):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return json.loads(completed.stdout)

    def test_analyze_placeholder_returns_score(self):
        result = self.run_script(
            {
                "action": "analyze",
                "uploadId": "upload_test",
                "sourcePath": None,
                "width": 1200,
                "height": 1600,
            }
        )

        self.assertIn("score", result)
        self.assertIn("summary", result)
        self.assertEqual(result["metrics"]["width"], 1024)
        self.assertEqual(result["metrics"]["height"], 1024)

    def test_preview_writes_output_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"

            Image.new("RGB", (512, 512), color=(140, 90, 80)).save(source_path)

            result = self.run_script(
                {
                    "action": "preview",
                    "uploadId": "upload_test",
                    "preset": "professional",
                    "sourcePath": str(source_path),
                    "outputPath": str(output_path),
                    "watermarkText": "RizzUp Preview",
                }
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(Path(result["previewPath"]), output_path)
            self.assertGreater(result["width"], 0)
            self.assertGreater(result["height"], 0)


if __name__ == "__main__":
    unittest.main()
