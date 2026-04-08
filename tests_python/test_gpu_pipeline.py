import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gpu_pipeline.py"
SPEC = importlib.util.spec_from_file_location("gpu_pipeline", SCRIPT)
GPU_PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GPU_PIPELINE)


class GpuPipelineTests(unittest.TestCase):
    def run_script(self, payload, check=True):
        return subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

    def test_analyze_placeholder_returns_score(self):
        completed = self.run_script(
            {
                "action": "analyze",
                "uploadId": "upload_test",
                "sourcePath": None,
                "width": 1200,
                "height": 1600,
            }
        )
        completed.check_returncode()
        result = json.loads(completed.stdout)

        self.assertIn("score", result)
        self.assertIn("summary", result)
        self.assertEqual(result["metrics"]["width"], 1024)
        self.assertEqual(result["metrics"]["height"], 1024)

    def test_preview_writes_output_image_with_face_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"

            Image.new("RGB", (900, 1200), color=(140, 90, 80)).save(source_path)
            fake_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (340, 260),
                    "rightEye": (440, 260),
                    "noseTip": (390, 320),
                    "mouthCenter": (390, 370),
                },
                "debug": {
                    "rawFaces": [[280, 180, 220, 220]],
                    "rawEyes": [[30, 50, 40, 20], [130, 50, 40, 20]],
                },
            }
            with patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face):
                result = GPU_PIPELINE.handle_preview(
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
            self.assertEqual(result["width"], 512)
            self.assertEqual(result["height"], 640)
            self.assertEqual(result["identityContext"]["embeddingSize"], 48)
            with Image.open(output_path) as preview:
                self.assertGreater(preview.size[1], preview.size[0])

    def test_preview_returns_structured_face_error_when_no_face_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"

            Image.new("RGB", (512, 512), color=(40, 80, 140)).save(source_path)

            completed = self.run_script(
                {
                    "action": "preview",
                    "uploadId": "upload_test",
                    "preset": "professional",
                    "sourcePath": str(source_path),
                    "outputPath": str(output_path),
                    "watermarkText": "RizzUp Preview",
                },
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            payload = json.loads(completed.stderr.strip().splitlines()[-1])
            self.assertEqual(payload["code"], "FACE_NOT_DETECTED")
            self.assertEqual(payload["message"], GPU_PIPELINE.FACE_NOT_DETECTED_MESSAGE)
            self.assertFalse(payload["retryable"])

    def test_skin_cleanup_stays_subtle(self):
        width, height = 320, 320
        source = Image.new("RGB", (width, height), color=(115, 100, 92))
        pixels = source.load()
        for y in range(90, 230):
            for x in range(90, 230):
                pixels[x, y] = (176 + (x % 6), 139 + (y % 5), 122 + ((x + y) % 4))

        face = {
            "box": {"x": 90, "y": 90, "w": 140, "h": 140},
            "landmarks": {},
        }
        cleaned = GPU_PIPELINE.subtle_skin_cleanup(source, face)
        source_array = np.asarray(source.crop((90, 90, 230, 230)), dtype=np.float32)
        cleaned_array = np.asarray(cleaned.crop((90, 90, 230, 230)), dtype=np.float32)
        mean_delta = abs(cleaned_array - source_array).mean()
        self.assertLess(mean_delta, 8.0)


if __name__ == "__main__":
    unittest.main()
