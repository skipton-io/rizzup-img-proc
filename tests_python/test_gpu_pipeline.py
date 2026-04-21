import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gpu_pipeline.py"
SPEC = importlib.util.spec_from_file_location("gpu_pipeline", SCRIPT)
GPU_PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GPU_PIPELINE)


class GpuPipelineTests(unittest.TestCase):
    def create_logo(self, path: Path, size=(600, 180), color=(255, 255, 255, 220)):
        logo = Image.new("RGBA", size, (0, 0, 0, 0))
        for y in range(size[1]):
            for x in range(size[0]):
                logo.putpixel((x, y), color)
        logo.save(path)

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
        self.assertEqual(result["metrics"]["width"], 100)
        self.assertEqual(result["metrics"]["height"], 100)

    def test_validate_upload_returns_cached_face_detection_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"

            Image.new("RGB", (900, 1200), color=(140, 90, 80)).save(source_path)
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

            with patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face):
                result = GPU_PIPELINE.handle_validate_upload(
                    {
                        "action": "validate_upload",
                        "uploadId": "upload_test",
                        "sourcePath": str(source_path),
                    }
                )

            self.assertEqual(result["faceDetection"]["box"], fake_face["box"])
            self.assertFalse(result["faceDetection"]["rotatedToPortrait"])

    def test_analyze_uses_cached_upload_face_detection_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"

            Image.new("RGB", (3264, 2448), color=(140, 90, 80)).save(source_path)
            cached_face = {
                "box": {"x": 882, "y": 325, "w": 1634, "h": 1634},
                "landmarks": {
                    "leftEye": [1079, 1790],
                    "rightEye": [1951, 1015],
                    "noseTip": [1699, 1272],
                    "mouthCenter": [1699, 1599],
                },
                "debug": {"rawFaces": [[882, 325, 1634, 1634]], "rawEyes": []},
                "rotatedToPortrait": False,
                "rotationDegrees": 0,
            }

            with patch.object(
                GPU_PIPELINE,
                "detect_primary_face",
                side_effect=AssertionError("analyze should reuse cached upload-time face detection"),
            ):
                result = GPU_PIPELINE.handle_analyze(
                    {
                        "action": "analyze",
                        "uploadId": "upload_cached_analyze",
                        "sourcePath": str(source_path),
                        "analysisMaxSize": 100,
                        "faceDetection": cached_face,
                    }
                )

            self.assertEqual(result["metrics"]["width"], 100)
            self.assertEqual(result["metrics"]["height"], 75)

    def test_preview_writes_output_image_with_face_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (900, 1200), color=(140, 90, 80)).save(source_path)
            self.create_logo(logo_path)
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
                        "watermarkLogoPath": str(logo_path),
                        "faceDetection": {
                            **fake_face,
                            "landmarks": {
                                key: list(value) for key, value in fake_face["landmarks"].items()
                            },
                            "rotatedToPortrait": False,
                        },
                    }
                )

            self.assertTrue(output_path.exists())
            self.assertEqual(Path(result["previewPath"]).resolve(), output_path.resolve())
            self.assertEqual(result["width"], 512)
            self.assertEqual(result["height"], 640)
            self.assertEqual(result["identityContext"]["embeddingSize"], 48)
            self.assertFalse(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "deterministic-enhancement")
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

    def test_firered_generation_falls_back_to_deterministic_when_model_load_fails(self):
        image = Image.new("RGB", (512, 512), color=(120, 100, 90))
        face = {
            "box": {"x": 120, "y": 120, "w": 200, "h": 200},
            "landmarks": {
                "leftEye": (180, 200),
                "rightEye": (260, 200),
                "noseTip": (220, 250),
                "mouthCenter": (220, 290),
            },
            "debug": {"rawFaces": [], "rawEyes": []},
        }

        with patch.object(
            GPU_PIPELINE,
            "generate_with_firered",
            side_effect=RuntimeError("missing weights"),
        ):
            generated, meta = GPU_PIPELINE.apply_identity_preserving_generation(
                image,
                face,
                {
                    "uploadId": "upload_firered_fallback",
                    "preset": "professional",
                    "fireRedEnabled": True,
                },
            )

        self.assertEqual(generated.size, image.size)
        self.assertFalse(meta["identityGenerationUsed"])
        self.assertEqual(meta["identityGenerationMode"], "deterministic-enhancement")
        self.assertEqual(meta["identityFallbackReason"], "missing weights")

    def test_firered_generation_uses_zimage_mode_when_enabled(self):
        image = Image.new("RGB", (512, 512), color=(120, 100, 90))
        face = {
            "box": {"x": 120, "y": 120, "w": 200, "h": 200},
            "landmarks": {
                "leftEye": (180, 200),
                "rightEye": (260, 200),
                "noseTip": (220, 250),
                "mouthCenter": (220, 290),
            },
            "debug": {"rawFaces": [], "rawEyes": []},
        }
        generated_image = Image.new("RGB", (512, 512), color=(150, 110, 100))

        with patch.object(GPU_PIPELINE, "generate_with_firered", return_value=generated_image):
            generated, meta = GPU_PIPELINE.apply_identity_preserving_generation(
                image,
                face,
                {
                    "uploadId": "upload_firered_makeup",
                    "preset": "professional",
                    "fireRedEnabled": True,
                    "fireRedPrompt": "Beautify this image",
                },
            )

        self.assertEqual(generated.size, generated_image.size)
        self.assertTrue(meta["identityGenerationUsed"])
        self.assertEqual(meta["identityGenerationMode"], "z-image-turbo")
        self.assertIsNone(meta["identityFallbackReason"])

    def test_firered_generation_does_not_run_when_disabled(self):
        image = Image.new("RGB", (512, 512), color=(120, 100, 90))
        face = {
            "box": {"x": 120, "y": 120, "w": 200, "h": 200},
            "landmarks": {
                "leftEye": (180, 200),
                "rightEye": (260, 200),
                "noseTip": (220, 250),
                "mouthCenter": (220, 290),
            },
            "debug": {"rawFaces": [], "rawEyes": []},
        }

        with patch.object(
            GPU_PIPELINE,
            "generate_with_firered",
            side_effect=AssertionError("FireRed should not run when disabled"),
        ):
            generated, meta = GPU_PIPELINE.apply_identity_preserving_generation(
                image,
                face,
                {
                    "uploadId": "upload_firered_disabled",
                    "preset": "professional",
                    "fireRedEnabled": False,
                },
            )

        self.assertEqual(generated.size, image.size)
        self.assertFalse(meta["identityGenerationUsed"])
        self.assertEqual(meta["identityGenerationMode"], "deterministic-enhancement")
        self.assertIsNone(meta["identityFallbackReason"])

    def test_preview_uses_deterministic_enhancement_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (900, 1200), color=(125, 100, 90)).save(source_path)
            self.create_logo(logo_path)
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

            with patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_test",
                        "preset": "natural",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertFalse(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "deterministic-enhancement")
            self.assertIsNone(result["identityFallbackReason"])

    def test_preview_skips_removed_identity_generation_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (900, 1200), color=(140, 90, 80))
            source.save(source_path)
            self.create_logo(logo_path)
            fake_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (340, 260),
                    "rightEye": (440, 260),
                    "noseTip": (390, 320),
                    "mouthCenter": (390, 370),
                },
                "debug": {"rawFaces": [[280, 180, 220, 220]], "rawEyes": []},
                "rotatedToPortrait": False,
            }
            result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_unstable_identity",
                        "preset": "natural",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                        "faceDetection": fake_face,
                    }
                )

            self.assertFalse(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "deterministic-enhancement")
            self.assertIsNone(result["rejectedPreviewPath"])

    def test_preview_skips_removed_identity_quality_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (900, 1200), color=(140, 90, 80))
            source.save(source_path)
            self.create_logo(logo_path)
            fake_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (340, 260),
                    "rightEye": (440, 260),
                    "noseTip": (390, 320),
                    "mouthCenter": (390, 370),
                },
                "debug": {"rawFaces": [[280, 180, 220, 220]], "rawEyes": []},
                "rotatedToPortrait": False,
            }
            result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_threshold_fallback",
                        "preset": "natural",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                        "faceDetection": fake_face,
                    }
                )

            self.assertFalse(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "deterministic-enhancement")

    def test_preview_identity_error_mode_is_ignored_in_deterministic_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (900, 1200), color=(125, 100, 90)).save(source_path)
            self.create_logo(logo_path)
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

            with patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_test",
                        "preset": "natural",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                        "previewIdentityEnabled": True,
                        "previewIdentityFallbackMode": "error",
                    }
                )

            self.assertEqual(result["identityGenerationMode"], "deterministic-enhancement")

    def test_preview_runs_post_processing_after_deterministic_enhancement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (900, 1200), color=(140, 90, 80)).save(source_path)
            self.create_logo(logo_path)
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
                        "watermarkLogoPath": str(logo_path),
                        "faceDetection": {
                            **fake_face,
                            "landmarks": {
                                key: list(value) for key, value in fake_face["landmarks"].items()
                            },
                            "rotatedToPortrait": False,
                        },
                    }
                )

            correct_mock.assert_called_once()
            cleanup_mock.assert_called_once()
            background_mock.assert_called_once()
            framing_mock.assert_called_once()

    def test_preview_uses_cached_face_detection_without_redetecting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (900, 1200), color=(140, 90, 80)).save(source_path)
            self.create_logo(logo_path)
            cached_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": [340, 260],
                    "rightEye": [440, 260],
                    "noseTip": [390, 320],
                    "mouthCenter": [390, 370],
                },
                "debug": {"rawFaces": [[280, 180, 220, 220]], "rawEyes": []},
                "rotatedToPortrait": False,
            }

            with (
                patch.object(
                    GPU_PIPELINE,
                    "detect_primary_face",
                    side_effect=AssertionError("preview should not re-detect when cached upload face exists"),
                ),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        Image.open(source_path).convert("RGB"),
                        {
                            "identityGenerationUsed": False,
                            "identityGenerationMode": "deterministic-enhancement",
                            "identityFallbackReason": None,
                        },
                    ),
                ),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_cached_face",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                        "faceDetection": cached_face,
                    }
                )

            expected_face = GPU_PIPELINE.scale_face_detection(cached_face, 384 / 900, 512 / 1200)
            self.assertEqual(result["identityContext"]["faceBox"], expected_face["box"])

    def test_preview_detects_face_when_no_cached_face_detection_is_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (900, 1200), color=(140, 90, 80)).save(source_path)
            self.create_logo(logo_path)
            detected_face = {
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
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=detected_face) as detect_mock,
                patch.object(GPU_PIPELINE, "correct_lighting", side_effect=lambda image: image),
                patch.object(GPU_PIPELINE, "subtle_skin_cleanup", side_effect=lambda image, face: image),
                patch.object(GPU_PIPELINE, "improve_background", side_effect=lambda image, face: image),
                patch.object(
                    GPU_PIPELINE,
                    "optimize_framing",
                    side_effect=lambda image, face, target_size=(512, 640): image.resize(target_size),
                ),
                patch.object(GPU_PIPELINE, "add_logo_watermark", side_effect=lambda image, path, text: image),
            ):
                GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_uncached_face",
                        "preset": "natural",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            detect_mock.assert_called()

    def test_final_uses_cached_face_detection_without_redetecting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "final.png"

            Image.new("RGB", (1200, 1600), color=(140, 90, 80)).save(source_path)
            cached_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": [340, 260],
                    "rightEye": [440, 260],
                    "noseTip": [390, 320],
                    "mouthCenter": [390, 370],
                },
                "debug": {"rawFaces": [[280, 180, 220, 220]], "rawEyes": []},
                "rotatedToPortrait": False,
                "rotationDegrees": 0,
            }

            with (
                patch.object(
                    GPU_PIPELINE,
                    "detect_primary_face",
                    side_effect=AssertionError("final should not re-detect when cached upload face exists"),
                ),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        Image.open(source_path).convert("RGB"),
                        {
                            "identityGenerationUsed": False,
                            "identityGenerationMode": "deterministic-enhancement",
                            "identityFallbackReason": None,
                        },
                    ),
                ),
            ):
                result = GPU_PIPELINE.handle_final(
                    {
                        "action": "final",
                        "uploadId": "upload_cached_face_final",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "faceDetection": cached_face,
                    }
                )

            self.assertTrue(output_path.exists())
            self.assertGreaterEqual(result["width"], 1024)
            self.assertGreaterEqual(result["height"], 1280)

    def test_preview_does_not_rotate_when_cached_rotation_is_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (900, 1200), color=(140, 90, 80))
            source.putpixel((0, 0), (255, 0, 0))
            source.save(source_path)
            self.create_logo(logo_path)
            cached_face = {
                "box": {"x": 280, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": [340, 260],
                    "rightEye": [440, 260],
                    "noseTip": [390, 320],
                    "mouthCenter": [390, 370],
                },
                "debug": {"rawFaces": [[280, 180, 220, 220]], "rawEyes": []},
                "rotatedToPortrait": False,
                "rotationDegrees": 0,
            }
            captured = {}

            def fake_identity(image, face, request):
                del face, request
                captured["top_left"] = image.getpixel((0, 0))
                return image, {
                    "identityGenerationUsed": False,
                    "identityGenerationMode": "deterministic-enhancement",
                    "identityFallbackReason": None,
                }

            with (
                patch.object(
                    GPU_PIPELINE,
                    "detect_primary_face",
                    side_effect=AssertionError("cached rotation should avoid re-detection"),
                ),
                patch.object(GPU_PIPELINE, "apply_identity_preserving_generation", side_effect=fake_identity),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_rotation_zero",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                        "faceDetection": cached_face,
                    }
                )

            self.assertFalse(result["rotatedToPortrait"])
            self.assertEqual(result["rotationDegrees"], 0)
            self.assertGreater(captured["top_left"][0], captured["top_left"][1])
            self.assertGreater(captured["top_left"][0], captured["top_left"][2])

    def test_preview_rotates_landscape_source_to_portrait_before_processing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (1200, 900), color=(140, 90, 80)).save(source_path)
            self.create_logo(logo_path)
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
                if image.width > image.height:
                    raise GPU_PIPELINE.PipelineValidationError(
                        "FACE_NOT_DETECTED",
                        GPU_PIPELINE.FACE_NOT_DETECTED_MESSAGE,
                        retryable=False,
                    )
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
                            "identityGenerationMode": "photomaker",
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
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertTrue(result["rotatedToPortrait"])
            with Image.open(output_path) as preview:
                self.assertGreater(preview.size[1], preview.size[0])

    def test_preview_keeps_upright_landscape_source_without_rotation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (1200, 900), color=(140, 90, 80))
            source.putpixel((0, 0), (255, 0, 0))
            source.save(source_path)
            self.create_logo(logo_path)

            upright_face = {
                "box": {"x": 220, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (280, 260),
                    "rightEye": (380, 260),
                    "noseTip": (330, 320),
                    "mouthCenter": (330, 370),
                },
                "debug": {"rawFaces": [[220, 180, 220, 220]], "rawEyes": [[30, 50, 40, 20], [130, 50, 40, 20]]},
            }

            def fake_identity(image, face, request):
                self.assertLessEqual(max(image.size), 512)
                self.assertGreater(image.getpixel((0, 0))[0], image.getpixel((0, 0))[1])
                return image, {
                    "identityGenerationUsed": True,
                    "identityGenerationMode": "photomaker",
                    "identityFallbackReason": None,
                }

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=upright_face),
                patch.object(GPU_PIPELINE, "apply_identity_preserving_generation", side_effect=fake_identity),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_landscape_upright",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertFalse(result["rotatedToPortrait"])
            self.assertTrue(output_path.exists())

    def test_preview_picks_better_portrait_rotation_for_landscape_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (1200, 900), color=(140, 90, 80))
            source.putpixel((0, 0), (255, 0, 0))
            source.putpixel((1199, 0), (0, 0, 255))
            source.save(source_path)
            self.create_logo(logo_path)
            clockwise_marker = source.rotate(-90, expand=True).getpixel((0, 0))
            counterclockwise_marker = source.rotate(90, expand=True).getpixel((0, 0))

            clockwise_face = {
                "box": {"x": 220, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (280, 260),
                    "rightEye": (380, 260),
                    "noseTip": (330, 320),
                    "mouthCenter": (330, 370),
                },
                "debug": {"rawFaces": [[220, 180, 220, 220]], "rawEyes": [[30, 50, 40, 20], [130, 50, 40, 20]]},
            }
            counterclockwise_face = {
                "box": {"x": 220, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (280, 260),
                    "rightEye": (380, 260),
                    "noseTip": (330, 320),
                    "mouthCenter": (330, 370),
                },
                "debug": {"rawFaces": [[220, 180, 220, 220]], "rawEyes": []},
            }

            captured_top_left = {}

            def fake_detect(image, request):
                if image.size == source.size:
                    raise GPU_PIPELINE.PipelineValidationError(
                        "FACE_NOT_DETECTED",
                        GPU_PIPELINE.FACE_NOT_DETECTED_MESSAGE,
                        retryable=False,
                    )
                top_left = image.getpixel((0, 0))
                if top_left == clockwise_marker:
                    return clockwise_face
                if top_left == counterclockwise_marker:
                    return counterclockwise_face
                raise AssertionError(f"Unexpected orientation marker: {top_left}")

            def fake_identity(image, face, request):
                captured_top_left["pixel"] = image.getpixel((0, 0))
                generated = image.copy()
                generated.putpixel((0, 0), clockwise_marker)
                generated.putpixel((generated.width - 1, generated.height - 1), counterclockwise_marker)
                return generated, {
                    "identityGenerationUsed": True,
                    "identityGenerationMode": "photomaker",
                    "identityFallbackReason": None,
                }

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(GPU_PIPELINE, "apply_identity_preserving_generation", side_effect=fake_identity),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_landscape_choice",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertTrue(result["rotatedToPortrait"])

    def test_preview_rotates_sideways_portrait_source_before_processing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (900, 1200), color=(140, 90, 80))
            source.putpixel((0, 0), (255, 0, 0))
            source.putpixel((899, 0), (0, 0, 255))
            source.save(source_path)
            self.create_logo(logo_path)
            clockwise_marker = source.rotate(-90, expand=True).getpixel((0, 0))
            counterclockwise_marker = source.rotate(90, expand=True).getpixel((0, 0))

            upright_face = {
                "box": {"x": 260, "y": 180, "w": 620, "h": 620},
                "landmarks": {
                    "leftEye": (420, 360),
                    "rightEye": (720, 360),
                    "noseTip": (570, 490),
                    "mouthCenter": (570, 620),
                },
                "debug": {"rawFaces": [[260, 180, 620, 620]], "rawEyes": [[40, 40, 60, 40], [180, 40, 60, 40]]},
            }
            weak_face = {
                "box": {"x": 40, "y": 40, "w": 90, "h": 90},
                "landmarks": {
                    "leftEye": (70, 76),
                    "rightEye": (100, 76),
                    "noseTip": (85, 92),
                    "mouthCenter": (85, 108),
                },
                "debug": {"rawFaces": [[40, 40, 90, 90]], "rawEyes": []},
            }

            captured_top_left = {}

            def fake_detect(image, request):
                top_left = image.getpixel((0, 0))
                if image.size == source.size:
                    return weak_face
                if top_left == clockwise_marker:
                    return weak_face
                if top_left == counterclockwise_marker:
                    return upright_face
                raise AssertionError(f"Unexpected orientation marker: {top_left}")

            def fake_identity(image, face, request):
                captured_top_left["pixel"] = image.getpixel((0, 0))
                generated = image.copy()
                generated.putpixel((0, 0), counterclockwise_marker)
                generated.putpixel((generated.width - 1, generated.height - 1), clockwise_marker)
                return generated, {
                    "identityGenerationUsed": True,
                    "identityGenerationMode": "photomaker",
                    "identityFallbackReason": None,
                }

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(GPU_PIPELINE, "apply_identity_preserving_generation", side_effect=fake_identity),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_portrait_sideways",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertTrue(result["rotatedToPortrait"])
            self.assertEqual(result["rotationDegrees"], 270)

    def test_preview_rotates_upside_down_landscape_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (1200, 900), color=(140, 90, 80))
            source.putpixel((0, 0), (255, 0, 0))
            source.putpixel((1199, 0), (0, 0, 255))
            source.save(source_path)
            self.create_logo(logo_path)
            clockwise_marker = source.rotate(-90, expand=True).getpixel((0, 0))
            counterclockwise_marker = source.rotate(90, expand=True).getpixel((0, 0))

            upside_down_face = {
                "box": {"x": 220, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (280, 320),
                    "rightEye": (380, 320),
                    "noseTip": (330, 270),
                    "mouthCenter": (330, 240),
                },
                "debug": {"rawFaces": [[220, 180, 220, 220]], "rawEyes": [[30, 50, 40, 20], [130, 50, 40, 20]]},
            }
            rotated_face = {
                "box": {"x": 220, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (280, 260),
                    "rightEye": (380, 260),
                    "noseTip": (330, 320),
                    "mouthCenter": (330, 370),
                },
                "debug": {"rawFaces": [[220, 180, 220, 220]], "rawEyes": [[30, 50, 40, 20], [130, 50, 40, 20]]},
            }

            captured_top_left = {}

            def fake_detect(image, request):
                top_left = image.getpixel((0, 0))
                if image.size == source.size:
                    return upside_down_face
                if top_left == clockwise_marker:
                    return rotated_face
                if top_left == counterclockwise_marker:
                    return rotated_face
                if top_left == (0, 255, 0):
                    return rotated_face
                raise AssertionError(f"Unexpected orientation marker: {top_left}")

            def fake_identity(image, face, request):
                captured_top_left["pixel"] = image.getpixel((0, 0))
                generated = image.copy()
                generated.putpixel((0, 0), clockwise_marker)
                generated.putpixel((generated.width - 1, generated.height - 1), (0, 255, 0))
                return generated, {
                    "identityGenerationUsed": True,
                    "identityGenerationMode": "photomaker",
                    "identityFallbackReason": None,
                }

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(GPU_PIPELINE, "apply_identity_preserving_generation", side_effect=fake_identity),
            ):
                result = GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_landscape_upside_down",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertTrue(result["rotatedToPortrait"])

    def test_preview_corrects_upside_down_generated_image_before_framing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (900, 1200), color=(140, 90, 80))
            source.putpixel((0, 0), (255, 0, 0))
            source.putpixel((899, 1199), (0, 255, 0))
            source.save(source_path)
            self.create_logo(logo_path)
            upside_down_generated = source.rotate(180, expand=True)

            source_face = {
                "box": {"x": 260, "y": 180, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (320, 260),
                    "rightEye": (420, 260),
                    "noseTip": (370, 320),
                    "mouthCenter": (370, 370),
                },
                "debug": {"rawFaces": [[260, 180, 220, 220]], "rawEyes": [[30, 50, 40, 20], [130, 50, 40, 20]]},
            }
            rotated_face = {
                "box": {"x": 420, "y": 800, "w": 220, "h": 220},
                "landmarks": {
                    "leftEye": (480, 940),
                    "rightEye": (580, 940),
                    "noseTip": (530, 900),
                    "mouthCenter": (530, 850),
                },
                "debug": {"rawFaces": [[420, 800, 220, 220]], "rawEyes": [[30, 50, 40, 20], [130, 50, 40, 20]]},
            }

            captured_top_left = {}

            def fake_detect(image, request):
                top_left = image.getpixel((0, 0))
                if top_left == (0, 255, 0):
                    return rotated_face
                if image.size == source.size or top_left == (255, 0, 0) or top_left == (140, 90, 80):
                    return source_face
                raise AssertionError(f"Unexpected orientation marker: {top_left}")

            def capture_framing(image, face, target_size=(512, 640)):
                del face, target_size
                captured_top_left["pixel"] = image.getpixel((0, 0))
                return image.resize((512, 640))

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(GPU_PIPELINE, "correct_lighting", side_effect=lambda image: image),
                patch.object(GPU_PIPELINE, "subtle_skin_cleanup", side_effect=lambda image, face: image),
                patch.object(GPU_PIPELINE, "improve_background", side_effect=lambda image, face: image),
                patch.object(GPU_PIPELINE, "optimize_framing", side_effect=capture_framing),
                patch.object(GPU_PIPELINE, "add_logo_watermark", side_effect=lambda image, path, text: image),
            ):
                GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_generated_upside_down",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertIn("pixel", captured_top_left)

    def test_preview_uses_source_face_box_for_framing_in_deterministic_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            output_path = temp_path / "preview.png"
            logo_path = temp_path / "logo.png"

            source = Image.new("RGB", (900, 1200), color=(140, 90, 80))
            source.putpixel((0, 0), (255, 0, 0))
            source.save(source_path)
            self.create_logo(logo_path)

            source_face = {
                "box": {"x": 40, "y": 900, "w": 120, "h": 120},
                "landmarks": {
                    "leftEye": (70, 940),
                    "rightEye": (120, 940),
                    "noseTip": (95, 980),
                    "mouthCenter": (95, 1010),
                },
                "debug": {"rawFaces": [[40, 900, 120, 120]], "rawEyes": [[10, 10, 20, 20], [60, 10, 20, 20]]},
            }
            captured_face_box = {}

            def fake_detect(image, request):
                del image, request
                return source_face

            def capture_framing(image, face, target_size=(512, 640)):
                del image, target_size
                captured_face_box["box"] = face["box"]
                return Image.new("RGB", (512, 640), color=(140, 90, 80))

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(GPU_PIPELINE, "correct_lighting", side_effect=lambda image: image),
                patch.object(GPU_PIPELINE, "subtle_skin_cleanup", side_effect=lambda image, face: image),
                patch.object(GPU_PIPELINE, "improve_background", side_effect=lambda image, face: image),
                patch.object(GPU_PIPELINE, "optimize_framing", side_effect=capture_framing),
                patch.object(GPU_PIPELINE, "add_logo_watermark", side_effect=lambda image, path, text: image),
            ):
                GPU_PIPELINE.handle_preview(
                    {
                        "action": "preview",
                        "uploadId": "upload_generated_face_box",
                        "preset": "professional",
                        "sourcePath": str(source_path),
                        "outputPath": str(output_path),
                        "watermarkText": "RizzUp Preview",
                        "watermarkLogoPath": str(logo_path),
                    }
                )

            self.assertEqual(captured_face_box["box"], {"x": 17, "y": 384, "w": 51, "h": 51})

    def test_final_reuses_preview_transform_pipeline_without_watermark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.png"
            preview_path = temp_path / "preview.png"
            final_path = temp_path / "final.png"
            logo_path = temp_path / "logo.png"

            Image.new("RGB", (1200, 900), color=(140, 90, 80)).save(source_path)
            self.create_logo(logo_path)
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

            def fake_detect(image, request):
                if image.width > image.height:
                    raise GPU_PIPELINE.PipelineValidationError(
                        "FACE_NOT_DETECTED",
                        GPU_PIPELINE.FACE_NOT_DETECTED_MESSAGE,
                        retryable=False,
                    )
                return fake_face

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        rotated_source,
                        {
                            "identityGenerationUsed": True,
                            "identityGenerationMode": "photomaker",
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
                        "watermarkLogoPath": str(logo_path),
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
            with Image.open(preview_path) as preview_image, Image.open(final_path) as final_image:
                self.assertGreaterEqual(final_image.width, 1024)
                self.assertGreaterEqual(final_image.height, 1280)
                self.assertGreater(final_image.width, preview_image.width)
                self.assertGreater(final_image.height, preview_image.height)
                preview_pixels = np.asarray(preview_image)
                final_pixels = np.asarray(final_image.resize(preview_image.size, Image.Resampling.LANCZOS))
                self.assertGreater(np.abs(preview_pixels.astype(np.int16) - final_pixels.astype(np.int16)).mean(), 0.5)

    def test_add_logo_watermark_scales_logo_within_preview_bounds(self):
        image = Image.new("RGB", (512, 640), color=(90, 120, 150))

        with tempfile.TemporaryDirectory() as temp_dir:
            logo_path = Path(temp_dir) / "logo.png"
            self.create_logo(logo_path, size=(1200, 600))

            watermarked = GPU_PIPELINE.add_logo_watermark(
                image,
                str(logo_path),
                "RizzUp Preview",
            )

        diff = ImageChops.difference(image.convert("RGB"), watermarked.convert("RGB"))
        bbox = diff.getbbox()

        self.assertIsNotNone(bbox)
        left, top, right, bottom = bbox
        self.assertLessEqual(right - left, int(image.width * 0.5))
        self.assertLessEqual(bottom - top, int(image.height * 0.3))

    def test_add_logo_watermark_falls_back_to_text_when_logo_is_missing(self):
        image = Image.new("RGB", (512, 640), color=(90, 120, 150))

        watermarked = GPU_PIPELINE.add_logo_watermark(
            image,
            "/path/to/missing-logo.png",
            "RizzUp Preview",
        )

        diff = ImageChops.difference(image.convert("RGB"), watermarked.convert("RGB"))
        self.assertIsNotNone(diff.getbbox())

    def test_preview_requires_a_source_image_when_source_path_is_missing(self):
        completed = self.run_script(
            {
                "action": "preview",
                "uploadId": "upload_missing_source",
                "preset": "professional",
                "sourcePath": None,
                "outputPath": "preview.png",
                "watermarkText": "RizzUp Preview",
            },
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads(completed.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["code"], "SOURCE_IMAGE_REQUIRED")
        self.assertFalse(payload["retryable"])

    def test_preview_requires_a_source_image_when_source_path_does_not_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "preview.png"
            missing_path = Path(temp_dir) / "missing.png"

            completed = self.run_script(
                {
                    "action": "preview",
                    "uploadId": "upload_missing_file",
                    "preset": "professional",
                    "sourcePath": str(missing_path),
                    "outputPath": str(output_path),
                    "watermarkText": "RizzUp Preview",
                },
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads(completed.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["code"], "SOURCE_IMAGE_REQUIRED")
        self.assertFalse(payload["retryable"])


if __name__ == "__main__":
    unittest.main()
