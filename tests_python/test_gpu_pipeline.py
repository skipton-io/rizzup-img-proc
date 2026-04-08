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
            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        Image.open(source_path).convert("RGB"),
                        {
                            "identityGenerationUsed": True,
                            "identityGenerationMode": "instantid",
                            "identityFallbackReason": None,
                        },
                    ),
                ),
            ):
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
            self.assertTrue(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "instantid")
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

    def test_debug_log_handles_numpy_scalar_values(self):
        payload = {
            "box": {"x": np.int32(10), "y": np.int32(20), "w": np.int32(30), "h": np.int32(40)},
            "landmarks": {
                "leftEye": (np.int32(1), np.int32(2)),
                "rightEye": (np.int32(3), np.int32(4)),
            },
            "rawEyes": np.array([[np.int32(5), np.int32(6), np.int32(7), np.int32(8)]], dtype=np.int32),
        }

        sanitized = GPU_PIPELINE.sanitize_for_json(payload)

        self.assertEqual(
            sanitized,
            {
                "box": {"x": 10, "y": 20, "w": 30, "h": 40},
                "landmarks": {"leftEye": [1, 2], "rightEye": [3, 4]},
                "rawEyes": [[5, 6, 7, 8]],
            },
        )

    def test_preview_identity_fallback_returns_heuristic_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"

            Image.new("RGB", (900, 1200), color=(125, 100, 90)).save(source_path)
            fake_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (340, 260),
                    "rightEye": (440, 260),
                    "noseTip": (390, 320),
                    "mouthCenter": (390, 370),
                },
                "debug": {"rawFaces": [[280, 180, 220, 220]], "rawEyes": []},
            }

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "get_instantid_generator",
                    side_effect=RuntimeError("missing InstantID checkpoints"),
                ),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_test",
                        "preset": "natural",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "previewIdentityEnabled": True,
                        "previewIdentityFallbackMode": "heuristic",
                    }
                )

            self.assertFalse(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "heuristic-fallback")
            self.assertIn("InstantID preview generation unavailable", result["identityFallbackReason"])

    def test_preview_identity_failure_can_return_structured_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"

            Image.new("RGB", (900, 1200), color=(125, 100, 90)).save(source_path)
            fake_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (340, 260),
                    "rightEye": (440, 260),
                    "noseTip": (390, 320),
                    "mouthCenter": (390, 370),
                },
                "debug": {"rawFaces": [[280, 180, 220, 220]], "rawEyes": []},
            }

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "get_instantid_generator",
                    side_effect=RuntimeError("missing InstantID checkpoints"),
                ),
            ):
                with self.assertRaises(GPU_PIPELINE.PipelineValidationError) as exc_info:
                    GPU_PIPELINE.handle_preview(
                        {
                            "action": "preview",
                            "uploadId": "upload_test",
                            "preset": "natural",
                            "sourcePath": str(source_path),
                            "outputPath": str(output_path),
                            "watermarkText": "RizzUp Preview",
                            "previewIdentityEnabled": True,
                            "previewIdentityFallbackMode": "error",
                        }
                    )

            self.assertEqual(exc_info.exception.code, GPU_PIPELINE.IDENTITY_GENERATION_ERROR_CODE)

    def test_preview_runs_post_processing_after_identity_generation(self):
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

            original_correct = GPU_PIPELINE.correct_lighting
            original_cleanup = GPU_PIPELINE.subtle_skin_cleanup
            original_background = GPU_PIPELINE.improve_background
            original_framing = GPU_PIPELINE.optimize_framing

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        Image.open(source_path).convert("RGB"),
                        {
                            "identityGenerationUsed": True,
                            "identityGenerationMode": "instantid",
                            "identityFallbackReason": None,
                        },
                    ),
                ),
                patch.object(GPU_PIPELINE, "correct_lighting", wraps=original_correct) as correct_mock,
                patch.object(GPU_PIPELINE, "subtle_skin_cleanup", wraps=original_cleanup) as cleanup_mock,
                patch.object(GPU_PIPELINE, "improve_background", wraps=original_background) as background_mock,
                patch.object(GPU_PIPELINE, "optimize_framing", wraps=original_framing) as framing_mock,
            ):
                GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_test",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                    }
                )

            correct_mock.assert_called_once()
            cleanup_mock.assert_called_once()
            background_mock.assert_called_once()
            framing_mock.assert_called_once()

    def test_preview_rotates_landscape_source_to_portrait_before_processing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"

            Image.new("RGB", (1200, 900), color=(140, 90, 80)).save(source_path)
            fake_face = {
                "box": {"x": 220, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (280, 260),
                    "rightEye": (380, 260),
                    "noseTip": (330, 320),
                    "mouthCenter": (330, 370),
                },
                "debug": {"rawFaces": [[220, 180, 220, 220]], "rawEyes": []},
            }

            def fake_detect(image, request):
                self.assertGreater(image.height, image.width)
                return fake_face

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        Image.open(source_path).convert("RGB").rotate(90, expand=True),
                        {
                            "identityGenerationUsed": True,
                            "identityGenerationMode": "instantid",
                            "identityFallbackReason": None,
                        },
                    ),
                ),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_landscape",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                    }
                )

            self.assertTrue(result["rotatedToPortrait"])
            with Image.open(output_path) as preview:
                self.assertGreater(preview.size[1], preview.size[0])

    def test_final_reuses_preview_transform_pipeline_without_watermark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            preview_path = temp_path / "preview.png"
            final_path = temp_path / "final.png"

            Image.new("RGB", (1200, 900), color=(140, 90, 80)).save(source_path)
            fake_face = {
                "box": {"x": 220, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (280, 260),
                    "rightEye": (380, 260),
                    "noseTip": (330, 320),
                    "mouthCenter": (330, 370),
                },
                "debug": {"rawFaces": [[220, 180, 220, 220]], "rawEyes": []},
            }

            rotated_source = Image.open(source_path).convert("RGB").rotate(90, expand=True)

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        rotated_source,
                        {
                            "identityGenerationUsed": True,
                            "identityGenerationMode": "instantid",
                            "identityFallbackReason": None,
                        },
                    ),
                ),
            ):
                preview_result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_match",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(preview_path),
                        "watermarkText": "RizzUp Preview",
                    }
                )
                final_result = GPU_PIPELINE.handle_final(
                    {
                        "action": "final",
                        "uploadId": "upload_match",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(final_path),
                    }
                )

            self.assertTrue(preview_result["rotatedToPortrait"])
            self.assertTrue(final_result["rotatedToPortrait"])
            self.assertEqual(final_result["identityGenerationMode"], preview_result["identityGenerationMode"])
            self.assertEqual(final_result["width"], preview_result["width"] * 2)
            self.assertEqual(final_result["height"], preview_result["height"] * 2)
            with Image.open(preview_path) as preview_image, Image.open(final_path) as final_image:
                self.assertEqual(final_image.size, (preview_image.width * 2, preview_image.height * 2))
                preview_pixels = np.asarray(preview_image)
                final_pixels = np.asarray(final_image.resize(preview_image.size, Image.Resampling.LANCZOS))
                self.assertGreater(np.abs(preview_pixels.astype(np.int16) - final_pixels.astype(np.int16)).mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
