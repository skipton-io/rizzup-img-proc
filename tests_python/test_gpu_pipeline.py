import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        Image.open(source_path).convert("RGB"),
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
            self.assertEqual(Path(result["previewPath"]), output_path)
            self.assertEqual(result["width"], 512)
            self.assertEqual(result["height"], 640)
            self.assertEqual(result["identityContext"]["embeddingSize"], 48)
            self.assertTrue(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "photomaker")
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

    def test_photomaker_generator_passes_id_embeds_for_v2(self):
        generator = object.__new__(GPU_PIPELINE.PhotoMakerGenerator)
        generator.face_detector = object()
        generator.device = GPU_PIPELINE.torch.device("cpu")
        generator.dtype = GPU_PIPELINE.torch.float32
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 384), color=(120, 90, 80))]
        generator.pipe = MagicMock(return_value=mock_result)

        image = Image.new("RGB", (512, 384), color=(140, 100, 90))
        settings = {
            "steps": 30,
            "guidanceScale": 4.5,
            "startMergeStep": 10,
            "blendStrength": 0.35,
            "triggerWord": "img",
        }

        with patch("photomaker.analyze_faces", return_value=[{"embedding": np.ones((512,), dtype=np.float32), "bbox": [0, 0, 128, 128]}]):
            result = GPU_PIPELINE.PhotoMakerGenerator.generate(
                generator,
                image,
                settings,
                "portrait photo of a person img, test",
                "bad anatomy",
            )

        self.assertEqual(result.size, (512, 384))
        self.assertTrue(generator.pipe.called)
        call_kwargs = generator.pipe.call_args.kwargs
        self.assertIn("id_embeds", call_kwargs)
        self.assertEqual(tuple(call_kwargs["id_embeds"].shape), (1, 512))

    def test_preview_identity_fallback_returns_heuristic_metadata(self):
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

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "get_photomaker_generator",
                    side_effect=RuntimeError("missing PhotoMaker checkpoint"),
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
                        "watermarkLogoPath": str(logo_path),
                        "previewIdentityEnabled": True,
                        "previewIdentityFallbackMode": "heuristic",
                    }
                )

            self.assertFalse(result["identityGenerationUsed"])
            self.assertEqual(result["identityGenerationMode"], "heuristic-fallback")
            self.assertIn("PhotoMaker preview generation unavailable", result["identityFallbackReason"])

    def test_preview_falls_back_when_identity_generation_face_region_drifts_too_far(self):
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
            unstable_generated = Image.new("RGB", (384, 512), color=(20, 240, 20))

            with patch.object(
                GPU_PIPELINE,
                "apply_identity_preserving_generation",
                return_value=(
                    unstable_generated,
                    {
                        "identityGenerationUsed": True,
                        "identityGenerationMode": "photomaker",
                        "identityFallbackReason": None,
                    },
                ),
            ):
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
            self.assertEqual(result["identityGenerationMode"], "heuristic-fallback")
            self.assertIn("looked unstable", result["identityFallbackReason"])

    def test_preview_falls_back_when_quality_delta_crosses_threshold(self):
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
            generated = Image.new("RGB", (384, 512), color=(140, 90, 80))

            with (
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        generated,
                        {
                            "identityGenerationUsed": True,
                        "identityGenerationMode": "photomaker",
                            "identityFallbackReason": None,
                        },
                    ),
                ),
                patch.object(GPU_PIPELINE, "compute_face_region_delta", return_value=0.1496),
            ):
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
            self.assertEqual(result["identityGenerationMode"], "heuristic-fallback")

    def test_preview_identity_failure_can_return_structured_error(self):
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

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", return_value=fake_face),
                patch.object(
                    GPU_PIPELINE,
                    "get_photomaker_generator",
                    side_effect=RuntimeError("missing PhotoMaker checkpoint"),
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
                            "watermarkLogoPath": str(logo_path),
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
                patch.object(
                    GPU_PIPELINE,
                    "apply_identity_preserving_generation",
                    return_value=(
                        Image.open(source_path).convert("RGB"),
                        {
                            "identityGenerationUsed": True,
                            "identityGenerationMode": "photomaker",
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
                            "identityGenerationUsed": True,
                            "identityGenerationMode": "photomaker",
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

            def fake_identity(image, face, request):
                return upside_down_generated, {
                    "identityGenerationUsed": True,
                    "identityGenerationMode": "photomaker",
                    "identityFallbackReason": None,
                }

            def capture_framing(image, face, target_size=(512, 640)):
                del face, target_size
                captured_top_left["pixel"] = image.getpixel((0, 0))
                return image.resize((512, 640))

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(GPU_PIPELINE, "apply_identity_preserving_generation", side_effect=fake_identity),
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

            self.assertEqual(captured_top_left["pixel"], (255, 0, 0))

    def test_preview_uses_generated_face_box_for_framing_when_better(self):
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
            generated_face = {
                "box": {"x": 250, "y": 180, "w": 320, "h": 320},
                "landmarks": {
                    "leftEye": (340, 280),
                    "rightEye": (460, 280),
                    "noseTip": (400, 360),
                    "mouthCenter": (400, 430),
                },
                "debug": {"rawFaces": [[250, 180, 320, 320]], "rawEyes": [[20, 20, 40, 20], [120, 20, 40, 20]]},
            }

            captured_face_box = {}

            def fake_detect(image, request):
                top_left = image.getpixel((0, 0))
                if top_left == (255, 0, 0):
                    return generated_face
                if top_left == (140, 90, 80):
                    return source_face
                raise AssertionError("Unexpected generated image marker")

            def fake_identity(image, face, request):
                generated = image.copy()
                generated.putpixel((0, 0), (255, 0, 0))
                return generated, {
                    "identityGenerationUsed": True,
                    "identityGenerationMode": "photomaker",
                    "identityFallbackReason": None,
                }

            def capture_framing(image, face, target_size=(512, 640)):
                del image, target_size
                captured_face_box["box"] = face["box"]
                return Image.new("RGB", (512, 640), color=(140, 90, 80))

            with (
                patch.object(GPU_PIPELINE, "detect_primary_face", side_effect=fake_detect),
                patch.object(GPU_PIPELINE, "apply_identity_preserving_generation", side_effect=fake_identity),
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

            self.assertEqual(captured_face_box["box"], generated_face["box"])

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


if __name__ == "__main__":
    unittest.main()
