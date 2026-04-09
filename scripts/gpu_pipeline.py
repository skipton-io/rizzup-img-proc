import json
import importlib.util
import inspect
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


PRESET_TUNING = {
    "natural": {"brightness": 1.03, "contrast": 1.05, "saturation": 1.02, "temperature": 0.0},
    "professional": {"brightness": 1.02, "contrast": 1.08, "saturation": 0.92, "temperature": -0.03},
    "lifestyle": {"brightness": 1.06, "contrast": 1.10, "saturation": 1.14, "temperature": 0.04},
    "fitness": {"brightness": 1.04, "contrast": 1.18, "saturation": 1.10, "temperature": 0.02},
    "travel": {"brightness": 1.08, "contrast": 1.12, "saturation": 1.18, "temperature": 0.05},
}

FACE_NOT_DETECTED_MESSAGE = "No face detected. Please upload a clear photo with one visible face."
IDENTITY_GENERATION_ERROR_CODE = "IDENTITY_GENERATION_UNAVAILABLE"
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed, distorted face, extra limbs, duplicate features, "
    "waxy skin, oversmoothed skin, uncanny expression"
)

PRESET_PROMPTS = {
    "natural": "authentic natural-light dating profile portrait, realistic skin texture, clean separation, high detail",
    "professional": "polished professional dating profile portrait, flattering studio-style lighting, realistic skin texture, clean background",
    "lifestyle": "confident lifestyle dating profile portrait, candid premium editorial feel, realistic skin texture, subtle depth",
    "fitness": "athletic dating profile portrait, crisp detail, healthy skin texture, focused subject separation",
    "travel": "premium travel dating profile portrait, warm natural color, scenic but unobtrusive background, realistic skin texture",
}


_INSTANTID_GENERATOR = None
_INSTANTID_INIT_ERROR = None


class PipelineValidationError(Exception):
    def __init__(self, code, message, retryable=False, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self):
        payload = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def load_request():
    return json.loads(sys.stdin.read())


def debug_log(event, **details):
    payload = {"event": event, **sanitize_for_json(details)}
    sys.stderr.write(f"{json.dumps(payload)}\n")
    sys.stderr.flush()


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def now_ms():
    return int(time.perf_counter() * 1000)


def timed_call(event, func, **kwargs):
    started = now_ms()
    debug_log(f"{event}-start", **kwargs)
    result = func()
    debug_log(f"{event}-complete", durationMs=now_ms() - started, **kwargs)
    return result


def request_bool(request, key, default=False):
    value = request.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def request_float(request, key, default):
    value = request.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def request_int(request, key, default):
    value = request.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def open_or_placeholder(source_path):
    if source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return ImageOps.exif_transpose(Image.open(candidate)).convert("RGB")

    image = Image.new("RGB", (1024, 1024), color=(52, 86, 132))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 944, 944), outline=(235, 235, 235), width=6)
    draw.text((120, 130), "RizzUp preview source missing", fill=(255, 255, 255))
    return image


def open_source_image(source_path):
    if not source_path:
        raise PipelineValidationError(
            "SOURCE_IMAGE_REQUIRED",
            "Source image is required for preview generation.",
            retryable=False,
        )

    candidate = Path(source_path)
    if not candidate.exists():
        raise PipelineValidationError(
            "SOURCE_IMAGE_REQUIRED",
            "Source image is required for preview generation.",
            retryable=False,
            details={"sourcePath": str(candidate)},
        )

    return ImageOps.exif_transpose(Image.open(candidate)).convert("RGB")


def normalize_to_portrait(image):
    if image.height >= image.width:
        return image, False
    return image.rotate(90, expand=True), True


def rotate_image(image, rotation_degrees):
    normalized = int(rotation_degrees) % 360
    if normalized == 0:
        return image
    if normalized == 90:
        return image.rotate(-90, expand=True)
    if normalized == 180:
        return image.rotate(180, expand=True)
    if normalized == 270:
        return image.rotate(90, expand=True)
    raise ValueError(f"Unsupported rotation: {rotation_degrees}")


def score_face_candidate(face):
    box = face["box"]
    area_score = int(box["w"]) * int(box["h"])
    raw_eyes = face.get("debug", {}).get("rawEyes", [])
    eye_bonus = min(len(raw_eyes), 2) * 1_000_000
    return eye_bonus + area_score


def is_face_upside_down(face):
    landmarks = face.get("landmarks", {})
    left_eye = landmarks.get("leftEye")
    right_eye = landmarks.get("rightEye")
    mouth_center = landmarks.get("mouthCenter")
    if not left_eye or not right_eye or not mouth_center:
        return False

    eye_line_y = (int(left_eye[1]) + int(right_eye[1])) / 2
    return int(mouth_center[1]) < eye_line_y


def normalize_to_best_portrait(image, request):
    candidates = [
        ("original", 0, image),
        ("clockwise", 90, rotate_image(image, 90)),
        ("counterclockwise", 270, rotate_image(image, 270)),
        ("rotate180", 180, rotate_image(image, 180)),
    ]
    best = None
    failures = []

    for direction, rotation_degrees, candidate in candidates:
        try:
            face = detect_primary_face(candidate, request)
            score = score_orientation_candidate(face, None, candidate, None)
            debug_log(
                "portrait-normalization-candidate",
                direction=direction,
                rotationDegrees=rotation_degrees,
                score=score,
                selectedFace=face["box"],
                rawEyes=face.get("debug", {}).get("rawEyes", []),
            )
            if best is None or score > best["score"]:
                best = {
                    "image": candidate,
                    "face": face,
                    "score": score,
                    "direction": direction,
                    "rotationDegrees": rotation_degrees,
                }
        except PipelineValidationError as exc:
            failures.append(exc)
            debug_log(
                "portrait-normalization-candidate-failed",
                direction=direction,
                rotationDegrees=rotation_degrees,
                errorCode=exc.code,
                errorMessage=exc.message,
            )

    if best is not None:
        debug_log(
            "portrait-normalization-selected",
            direction=best["direction"],
            rotationDegrees=best["rotationDegrees"],
            score=best["score"],
        )
        return best["image"], best["rotationDegrees"] != 0, best["face"], best["rotationDegrees"]

    if failures:
        raise failures[0]
    raise PipelineValidationError("FACE_NOT_DETECTED", FACE_NOT_DETECTED_MESSAGE, retryable=False)


def face_center(box):
    return (
        float(box["x"]) + float(box["w"]) / 2.0,
        float(box["y"]) + float(box["h"]) / 2.0,
    )


def score_orientation_candidate(candidate_face, reference_face, candidate_image, expect_portrait):
    score = score_face_candidate(candidate_face)

    if not is_face_upside_down(candidate_face):
        score += 1_000_000

    if expect_portrait is not None and ((candidate_image.height >= candidate_image.width) == expect_portrait):
        score += 500_000

    if reference_face:
        candidate_center = face_center(candidate_face["box"])
        reference_center = face_center(reference_face["box"])
        distance = math.dist(candidate_center, reference_center)
        diagonal = math.hypot(candidate_image.width, candidate_image.height) or 1.0
        score += int(max(0.0, 1.0 - (distance / diagonal)) * 2_000_000)

    return score


def stabilize_generated_orientation(image, reference_face, request, expect_portrait):
    candidates = [
        ("original", image),
        ("rotate180", image.rotate(180, expand=True)),
    ]
    best = None

    for direction, candidate in candidates:
        try:
            detected_face = detect_primary_face(candidate, request)
            score = score_orientation_candidate(detected_face, reference_face, candidate, expect_portrait)
            debug_log(
                "generated-orientation-candidate",
                direction=direction,
                score=score,
                selectedFace=detected_face["box"],
                upright=not is_face_upside_down(detected_face),
            )
            if best is None or score > best["score"]:
                best = {"image": candidate, "score": score, "direction": direction, "face": detected_face}
        except PipelineValidationError as exc:
            debug_log(
                "generated-orientation-candidate-failed",
                direction=direction,
                errorCode=exc.code,
                errorMessage=exc.message,
            )

    if best is None:
        return image, reference_face

    debug_log("generated-orientation-selected", direction=best["direction"], score=best["score"])
    return best["image"], best["face"]


def cascade_classifier(custom_path, default_name):
    cascade_path = custom_path or str(Path(cv2.data.haarcascades) / default_name)
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        raise RuntimeError(f"Could not load cascade classifier: {cascade_path}")
    return classifier


def detect_primary_face(image, request):
    array = np.asarray(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    face_classifier = cascade_classifier(request.get("faceCascadePath"), "haarcascade_frontalface_default.xml")
    eye_classifier = cascade_classifier(request.get("eyeCascadePath"), "haarcascade_eye.xml")

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(72, 72),
    )
    raw_faces = [[int(value) for value in face.tolist()] for face in faces] if len(faces) else []
    debug_log("face-detect-raw", rawFaces=raw_faces, imageWidth=int(image.width), imageHeight=int(image.height))
    if len(faces) == 0:
        raise PipelineValidationError(
            "FACE_NOT_DETECTED",
            FACE_NOT_DETECTED_MESSAGE,
            retryable=False,
            details={
                "rawFaces": raw_faces,
                "rawEyes": [],
                "imageWidth": int(image.width),
                "imageHeight": int(image.height),
            },
        )

    faces = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
    x, y, w, h = [int(value) for value in faces[0]]
    face_roi = gray[y : y + h, x : x + w]
    eyes = eye_classifier.detectMultiScale(face_roi, scaleFactor=1.05, minNeighbors=3, minSize=(18, 18))
    raw_eyes = [[int(value) for value in eye.tolist()] for eye in eyes] if len(eyes) else []
    debug_log("eye-detect-raw", selectedFace={"x": x, "y": y, "w": w, "h": h}, rawEyes=raw_eyes)
    eyes = sorted(eyes, key=lambda item: item[2] * item[3], reverse=True)[:2]

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda item: item[0])
        left_eye = (
            int(x + int(eyes[0][0]) + int(eyes[0][2]) // 2),
            int(y + int(eyes[0][1]) + int(eyes[0][3]) // 2),
        )
        right_eye = (
            int(x + int(eyes[1][0]) + int(eyes[1][2]) // 2),
            int(y + int(eyes[1][1]) + int(eyes[1][3]) // 2),
        )
    else:
        left_eye = (int(x + w * 0.32), int(y + h * 0.4))
        right_eye = (int(x + w * 0.68), int(y + h * 0.4))

    result = {
        "box": {"x": x, "y": y, "w": w, "h": h},
        "landmarks": {
            "leftEye": left_eye,
            "rightEye": right_eye,
            "noseTip": (int(x + w * 0.5), int(y + h * 0.58)),
            "mouthCenter": (int(x + w * 0.5), int(y + h * 0.78)),
        },
        "debug": {
            "rawFaces": raw_faces,
            "rawEyes": raw_eyes,
        },
    }
    debug_log("face-detect-selected", selectedFace=result["box"], landmarks=result["landmarks"])
    return result


def build_identity_context(image, face):
    box = face["box"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    face_crop = np.asarray(image.crop((x, y, x + w, y + h)).resize((112, 112), Image.Resampling.LANCZOS)).astype(
        np.float32
    ) / 255.0
    tensor = torch.from_numpy(face_crop).permute(2, 0, 1).unsqueeze(0)
    embedding = torch.nn.functional.adaptive_avg_pool2d(tensor, (4, 4)).flatten().tolist()
    return {
        "embedding": [round(float(value), 6) for value in embedding],
        "faceBox": box,
        "landmarks": face["landmarks"],
    }


def normalize_cached_face(face):
    if not face:
        return None
    return {
        "box": {key: int(value) for key, value in face["box"].items()},
        "landmarks": {
            key: tuple(int(value) for value in point)
            for key, point in face["landmarks"].items()
        },
        "debug": {
            "rawFaces": [[int(value) for value in item] for item in face.get("debug", {}).get("rawFaces", [])],
            "rawEyes": [[int(value) for value in item] for item in face.get("debug", {}).get("rawEyes", [])],
        },
        "rotatedToPortrait": bool(face.get("rotatedToPortrait", False)),
        "rotationDegrees": int(face.get("rotationDegrees", 90 if face.get("rotatedToPortrait") else 0)) % 360,
    }


def load_cached_face_detection(request):
    cached_face = normalize_cached_face(request.get("faceDetection"))
    if not cached_face:
        return None
    return cached_face


def apply_cached_rotation(image, cached_face):
    rotation_degrees = int(cached_face.get("rotationDegrees", 90 if cached_face.get("rotatedToPortrait") else 0)) % 360
    if rotation_degrees:
        return rotate_image(image, rotation_degrees), True, rotation_degrees
    return image, False, 0


def build_preview_identity_settings(request):
    return {
        "enabled": request_bool(request, "previewIdentityEnabled", True),
        "fallbackMode": request.get("previewIdentityFallbackMode", "heuristic") or "heuristic",
        "cacheDir": request.get("previewIdentityCacheDir") or str(Path(".cache") / "instantid"),
        "pipelinePath": request.get("previewIdentityPipelinePath")
        or str(Path("third_party") / "InstantID" / "pipeline_stable_diffusion_xl_instantid.py"),
        "checkpointDir": request.get("previewIdentityCheckpointDir")
        or str(Path("third_party") / "InstantID" / "checkpoints"),
        "faceEncoderRoot": request.get("previewIdentityFaceEncoderRoot")
        or str(Path("third_party") / "InstantID"),
        "baseModel": request.get("previewIdentityBaseModel") or "stabilityai/stable-diffusion-xl-base-1.0",
        "promptTemplate": request.get("previewIdentityPromptTemplate"),
        "negativePrompt": request.get("previewIdentityNegativePrompt") or DEFAULT_NEGATIVE_PROMPT,
        "steps": request_int(request, "previewIdentitySteps", 30),
        "guidanceScale": request_float(request, "previewIdentityGuidanceScale", 4.5),
        "controlScale": request_float(request, "previewIdentityControlScale", 0.72),
        "adapterScale": request_float(request, "previewIdentityAdapterScale", 0.68),
        "blendStrength": request_float(request, "previewIdentityBlendStrength", 0.35),
    }


def build_identity_prompt(preset, settings):
    preset_prompt = PRESET_PROMPTS.get(preset, PRESET_PROMPTS["natural"])
    template = settings.get("promptTemplate")
    if template:
        return template.format(preset=preset, preset_prompt=preset_prompt)
    return preset_prompt


def _largest_face(face_infos):
    return sorted(face_infos, key=lambda item: item.get("bbox", [0, 0, 0, 0])[2] - item.get("bbox", [0, 0, 0, 0])[0], reverse=True)[0]


def _load_pipeline_module(pipeline_path):
    candidate = Path(pipeline_path)
    if not candidate.exists():
        raise FileNotFoundError(f"InstantID pipeline file not found: {candidate}")

    module_root = str(candidate.parent.resolve())
    if module_root not in sys.path:
        sys.path.insert(0, module_root)

    spec = importlib.util.spec_from_file_location("rizzup_instantid_pipeline", candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load InstantID pipeline module from {candidate}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstantIDGenerator:
    def __init__(self, settings):
        init_started = now_ms()
        from diffusers.models import ControlNetModel
        from insightface.app import FaceAnalysis

        module = timed_call(
            "instantid-module-load",
            lambda: _load_pipeline_module(settings["pipelinePath"]),
            pipelinePath=settings["pipelinePath"],
        )
        pipeline_class = getattr(module, "StableDiffusionXLInstantIDPipeline", None)
        draw_kps = getattr(module, "draw_kps", None)
        if pipeline_class is None or draw_kps is None:
            raise RuntimeError("InstantID pipeline module must expose StableDiffusionXLInstantIDPipeline and draw_kps")

        checkpoint_dir = Path(settings["checkpointDir"])
        controlnet_dir = checkpoint_dir / "ControlNetModel"
        adapter_path = checkpoint_dir / "ip-adapter.bin"
        if not controlnet_dir.exists():
            raise FileNotFoundError(f"InstantID ControlNet directory not found: {controlnet_dir}")
        if not adapter_path.exists():
            raise FileNotFoundError(f"InstantID adapter checkpoint not found: {adapter_path}")

        cache_dir = Path(settings["cacheDir"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.draw_kps = draw_kps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
        self.face_app = FaceAnalysis(name="antelopev2", root=settings["faceEncoderRoot"], providers=providers)
        timed_call(
            "instantid-face-app-prepare",
            lambda: self.face_app.prepare(ctx_id=0 if self.device.type == "cuda" else -1, det_size=(640, 640)),
            faceEncoderRoot=settings["faceEncoderRoot"],
            device=self.device.type,
        )

        controlnet = timed_call(
            "instantid-controlnet-load",
            lambda: ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=self.dtype, use_safetensors=True),
            controlnetDir=str(controlnet_dir),
        )
        self.pipe = timed_call(
            "instantid-pipeline-load",
            lambda: pipeline_class.from_pretrained(
                settings["baseModel"],
                controlnet=controlnet,
                torch_dtype=self.dtype,
                cache_dir=str(cache_dir),
                use_safetensors=True,
            ),
            baseModel=settings["baseModel"],
            cacheDir=str(cache_dir),
        )
        if hasattr(self.pipe, "to"):
            self.pipe = timed_call(
                "instantid-pipeline-to-device",
                lambda: self.pipe.to(self.device),
                device=self.device.type,
            )
        if hasattr(self.pipe, "load_ip_adapter_instantid"):
            timed_call(
                "instantid-adapter-load",
                lambda: self.pipe.load_ip_adapter_instantid(str(adapter_path)),
                adapterPath=str(adapter_path),
            )
        if hasattr(self.pipe, "set_ip_adapter_scale"):
            self.pipe.set_ip_adapter_scale(settings["adapterScale"])
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        debug_log(
            "instantid-generator-ready",
            durationMs=now_ms() - init_started,
            device=self.device.type,
            dtype=str(self.dtype),
            baseModel=settings["baseModel"],
        )

    def generate(self, image, settings, prompt, negative_prompt):
        generation_started = now_ms()
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        face_infos = timed_call(
            "instantid-face-analysis",
            lambda: self.face_app.get(bgr),
            imageWidth=int(image.width),
            imageHeight=int(image.height),
        )
        if not face_infos:
            raise RuntimeError("InstantID insightface encoder could not find a face in the source image")

        face_info = _largest_face(face_infos)
        face_emb = face_info.get("embedding")
        keypoints = face_info.get("kps")
        if face_emb is None or keypoints is None:
            raise RuntimeError("InstantID insightface encoder did not return embedding and keypoints")

        face_emb = torch.from_numpy(np.asarray(face_emb)).unsqueeze(0).to(self.device, dtype=self.dtype)
        keypoint_image = self.draw_kps(image, keypoints)

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_embeds": face_emb,
            "image": keypoint_image,
            "controlnet_conditioning_scale": settings["controlScale"],
            "ip_adapter_scale": settings["adapterScale"],
            "num_inference_steps": settings["steps"],
            "guidance_scale": settings["guidanceScale"],
        }
        signature = inspect.signature(self.pipe.__call__)
        if "width" in signature.parameters:
            call_kwargs["width"] = int(image.width)
        if "height" in signature.parameters:
            call_kwargs["height"] = int(image.height)

        debug_log(
            "instantid-pipeline-call-start",
            width=call_kwargs.get("width", int(image.width)),
            height=call_kwargs.get("height", int(image.height)),
            steps=settings["steps"],
            guidanceScale=settings["guidanceScale"],
            controlScale=settings["controlScale"],
            adapterScale=settings["adapterScale"],
        )
        result = self.pipe(**call_kwargs).images[0].convert("RGB")
        debug_log(
            "instantid-pipeline-call-complete",
            durationMs=now_ms() - generation_started,
            outputWidth=int(result.width),
            outputHeight=int(result.height),
        )
        blend_strength = min(max(settings["blendStrength"], 0.0), 1.0)
        base = image.resize(result.size, Image.Resampling.LANCZOS)
        blended = Image.blend(base, result, blend_strength)
        debug_log("instantid-generate-complete", totalDurationMs=now_ms() - generation_started, blendStrength=blend_strength)
        return blended


def get_instantid_generator(settings):
    global _INSTANTID_GENERATOR
    global _INSTANTID_INIT_ERROR

    if _INSTANTID_GENERATOR is not None:
        return _INSTANTID_GENERATOR
    if _INSTANTID_INIT_ERROR is not None:
        raise RuntimeError(_INSTANTID_INIT_ERROR)

    try:
        _INSTANTID_GENERATOR = InstantIDGenerator(settings)
        debug_log("instantid-init-complete", pipelinePath=settings["pipelinePath"], baseModel=settings["baseModel"])
        return _INSTANTID_GENERATOR
    except Exception as exc:
        _INSTANTID_INIT_ERROR = str(exc)
        debug_log("instantid-init-failed", error=str(exc), pipelinePath=settings["pipelinePath"])
        raise


def apply_identity_preserving_generation(image, face, request):
    del face
    settings = build_preview_identity_settings(request)
    if not settings["enabled"]:
        return image, {
            "identityGenerationUsed": False,
            "identityGenerationMode": "disabled",
            "identityFallbackReason": None,
        }

    prompt = build_identity_prompt(request.get("preset", "natural"), settings)
    negative_prompt = settings["negativePrompt"]
    debug_log(
        "instantid-generation-start",
        uploadId=request.get("uploadId"),
        pipelinePath=settings["pipelinePath"],
        baseModel=settings["baseModel"],
        steps=settings["steps"],
    )

    try:
        generator = get_instantid_generator(settings)
        generated = generator.generate(image, settings, prompt, negative_prompt)
        debug_log("instantid-generation-complete", uploadId=request.get("uploadId"), usedGpu=generator.device.type == "cuda")
        return generated, {
            "identityGenerationUsed": True,
            "identityGenerationMode": "instantid",
            "identityFallbackReason": None,
        }
    except Exception as exc:
        message = f"InstantID preview generation unavailable: {exc}"
        debug_log("instantid-generation-fallback", uploadId=request.get("uploadId"), fallbackMode=settings["fallbackMode"], error=message)
        if settings["fallbackMode"] == "error":
            raise PipelineValidationError(
                IDENTITY_GENERATION_ERROR_CODE,
                message,
                retryable=False,
                details={
                    "pipelinePath": settings["pipelinePath"],
                    "checkpointDir": settings["checkpointDir"],
                    "baseModel": settings["baseModel"],
                },
            )
        return image, {
            "identityGenerationUsed": False,
            "identityGenerationMode": "heuristic-fallback",
            "identityFallbackReason": message,
        }


def correct_lighting(image):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    adjusted_l = clahe.apply(l_channel)
    merged = cv2.merge((adjusted_l, a_channel, b_channel))
    corrected = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(corrected)


def subtle_skin_cleanup(image, face):
    box = face["box"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    array = np.asarray(image)
    roi = array[y : y + h, x : x + w]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    softened = cv2.bilateralFilter(roi, d=7, sigmaColor=24, sigmaSpace=16)
    blend = roi.copy()
    skin_pixels = skin_mask > 0
    blend[skin_pixels] = cv2.addWeighted(roi, 0.82, softened, 0.18, 0)[skin_pixels]
    output = array.copy()
    output[y : y + h, x : x + w] = blend
    return Image.fromarray(output)


def improve_background(image, face):
    array = np.asarray(image)
    box = face["box"]
    center_x = box["x"] + box["w"] / 2.0
    center_y = box["y"] + box["h"] * 1.15
    axes = (max(int(box["w"] * 1.2), 1), max(int(box["h"] * 1.9), 1))

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    cv2.ellipse(mask, (int(center_x), int(center_y)), axes, 0, 0, 360, 255, -1)
    background_mask = cv2.GaussianBlur(255 - mask, (0, 0), 9)
    background_mask = background_mask.astype(np.float32) / 255.0
    background_mask = background_mask[..., None]

    blurred = cv2.GaussianBlur(array, (0, 0), 3)
    desaturated = cv2.cvtColor(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    background = cv2.addWeighted(blurred, 0.86, desaturated, 0.14, 0)
    foreground = cv2.addWeighted(array, 0.94, cv2.GaussianBlur(array, (0, 0), 1.1), 0.06, 0)
    output = foreground * (1.0 - background_mask) + background * background_mask
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def optimize_framing(image, face, target_size=(512, 640)):
    target_width, target_height = target_size
    target_ratio = target_width / target_height
    box = face["box"]

    face_cx = box["x"] + box["w"] / 2.0
    face_top = box["y"]
    face_bottom = box["y"] + box["h"]
    desired_height = min(image.height, max(box["h"] * 3.8, 320))
    desired_width = desired_height * target_ratio

    top = max(0, face_top - box["h"] * 1.05)
    bottom = min(image.height, top + desired_height)
    top = max(0, bottom - desired_height)

    left = max(0, face_cx - desired_width / 2.0)
    right = min(image.width, left + desired_width)
    left = max(0, right - desired_width)

    crop = image.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))
    return crop.resize(target_size, Image.Resampling.LANCZOS)


def image_metrics(image):
    array = np.asarray(image).astype(np.float32) / 255.0
    luminance = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    gx = np.diff(luminance, axis=1, prepend=luminance[:, :1])
    gy = np.diff(luminance, axis=0, prepend=luminance[:1, :])

    brightness = float(luminance.mean())
    contrast = float(luminance.std())
    sharpness = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

    score = 64.0
    score += min(18.0, max(0.0, (brightness - 0.25) * 30.0))
    score += min(10.0, max(0.0, contrast * 20.0))
    score += min(10.0, max(0.0, sharpness * 40.0))
    score = max(35.0, min(96.0, score))

    return {
        "score": int(round(score)),
        "metrics": {
            "width": int(image.width),
            "height": int(image.height),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "sharpness": round(sharpness, 4),
        },
    }


def summarize(metrics):
    brightness = metrics["brightness"]
    sharpness = metrics["sharpness"]

    notes = [f"Image resolution is {metrics['width']}x{metrics['height']}."]
    if brightness < 0.32:
        notes.append("Exposure is a little dark for dating-profile use.")
    elif brightness > 0.78:
        notes.append("Highlights are bright and may need slight recovery.")
    else:
        notes.append("Exposure is in a healthy range for a clean headshot.")

    if sharpness < 0.06:
        notes.append("Subject detail looks soft, so a steadier source photo would help.")
    else:
        notes.append("Edge detail is strong enough for a polished preview.")

    notes.append("A tighter crop and uncluttered background should improve conversion.")
    return " ".join(notes)


def apply_preset_gpu(image, preset):
    tuning = PRESET_TUNING.get(preset, PRESET_TUNING["natural"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).to(device=device)
    tensor = tensor.permute(2, 0, 1)

    mean = tensor.mean(dim=0, keepdim=True)
    tensor = (tensor - mean) * tuning["contrast"] + mean
    tensor = tensor * tuning["brightness"]

    gray = tensor.mean(dim=0, keepdim=True)
    tensor = gray + (tensor - gray) * tuning["saturation"]

    tensor[0] = tensor[0] * (1.0 + tuning["temperature"])
    tensor[2] = tensor[2] * (1.0 - tuning["temperature"])

    height = tensor.shape[1]
    width = tensor.shape[2]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    vignette = 1.0 - 0.10 * torch.clamp(xx * xx + yy * yy, 0.0, 1.0)
    tensor = tensor * vignette.unsqueeze(0)
    tensor = tensor.clamp(0.0, 1.0)

    output = tensor.permute(1, 2, 0).mul(255.0).byte().cpu().numpy()
    return Image.fromarray(output, mode="RGB"), device.type == "cuda"


def add_text_watermark(image, watermark_text):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text = watermark_text or "RizzUp Preview"
    bbox = draw.textbbox((0, 0), text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = image.width - width - 24
    y = image.height - height - 24
    draw.rounded_rectangle(
        (x - 14, y - 10, x + width + 14, y + height + 10),
        radius=16,
        fill=(0, 0, 0, 150),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 225))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def resize_logo_to_fit(logo, max_width, max_height):
    width_ratio = max_width / max(logo.width, 1)
    height_ratio = max_height / max(logo.height, 1)
    scale = min(width_ratio, height_ratio)
    resized_width = max(1, int(round(logo.width * scale)))
    resized_height = max(1, int(round(logo.height * scale)))
    return logo.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def add_logo_watermark(image, watermark_logo_path, watermark_text):
    if watermark_logo_path:
        logo_path = Path(watermark_logo_path)
        if logo_path.exists():
            logo = Image.open(logo_path).convert("RGBA")
            max_logo_width = max(1, int(image.width * 0.5))
            max_logo_height = max(1, int(image.height * 0.3))
            logo = resize_logo_to_fit(logo, max_logo_width, max_logo_height)

            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            padding = max(16, int(min(image.width, image.height) * 0.03))
            x = image.width - logo.width - padding
            y = image.height - logo.height - padding

            logo_alpha = logo.getchannel("A").point(lambda alpha: int(alpha * 0.82))
            logo.putalpha(logo_alpha)
            overlay.alpha_composite(logo, (x, y))
            return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return add_text_watermark(image, watermark_text)


def run_face_aware_pipeline(request, add_preview_watermark):
    image = timed_call(
        "source-open",
        lambda: open_source_image(request.get("sourcePath")),
        uploadId=request.get("uploadId"),
        sourcePath=request.get("sourcePath"),
    )
    cached_face = load_cached_face_detection(request)
    if cached_face is not None:
        image, rotated_to_portrait, rotation_degrees = timed_call(
            "portrait-normalization-from-cache",
            lambda: apply_cached_rotation(image, cached_face),
            uploadId=request.get("uploadId"),
        )
        face = cached_face
    else:
        image, rotated_to_portrait, face, rotation_degrees = timed_call(
            "portrait-normalization",
            lambda: normalize_to_best_portrait(image, request),
            uploadId=request.get("uploadId"),
        )
    debug_log(
        "face-check-before",
        uploadId=request.get("uploadId"),
        preset=request.get("preset", "natural"),
        sourcePath=request.get("sourcePath"),
        rotatedToPortrait=rotated_to_portrait,
        rotationDegrees=rotation_degrees,
        imageWidth=int(image.width),
        imageHeight=int(image.height),
    )
    debug_log(
        "face-check-after",
        uploadId=request.get("uploadId"),
        accepted=True,
        rotatedToPortrait=rotated_to_portrait,
        rotationDegrees=rotation_degrees,
        selectedFace=face["box"],
        rawFaces=face.get("debug", {}).get("rawFaces", []),
        rawEyes=face.get("debug", {}).get("rawEyes", []),
    )
    identity_context = timed_call(
        "identity-context",
        lambda: build_identity_context(image, face),
        uploadId=request.get("uploadId"),
    )
    processed, identity_meta = timed_call(
        "identity-generation",
        lambda: apply_identity_preserving_generation(image, face, request),
        uploadId=request.get("uploadId"),
        preset=request.get("preset", "natural"),
    )
    processed, processed_face = timed_call(
        "generated-orientation-stabilization",
        lambda: stabilize_generated_orientation(processed, face, request, image.height >= image.width),
        uploadId=request.get("uploadId"),
    )
    face = processed_face or face
    processed = timed_call("lighting-correction", lambda: correct_lighting(processed), uploadId=request.get("uploadId"))
    processed = timed_call("skin-cleanup", lambda: subtle_skin_cleanup(processed, face), uploadId=request.get("uploadId"))
    processed = timed_call("background-improvement", lambda: improve_background(processed, face), uploadId=request.get("uploadId"))
    processed = timed_call("framing-optimization", lambda: optimize_framing(processed, face), uploadId=request.get("uploadId"))
    processed, used_gpu = timed_call(
        "preset-gpu",
        lambda: apply_preset_gpu(processed, request.get("preset", "natural")),
        uploadId=request.get("uploadId"),
        preset=request.get("preset", "natural"),
    )
    if add_preview_watermark:
        processed = timed_call(
            "watermark",
            lambda: add_logo_watermark(
                processed,
                request.get("watermarkLogoPath"),
                request.get("watermarkText", "RizzUp Preview"),
            ),
            uploadId=request.get("uploadId"),
        )

    return {
        "processed": processed,
        "usedGpu": used_gpu,
        "identityContext": identity_context,
        "identityMeta": identity_meta,
        "rotatedToPortrait": rotated_to_portrait,
        "rotationDegrees": rotation_degrees,
    }


def handle_analyze(request):
    image = open_or_placeholder(request.get("sourcePath"))
    if request.get("sourcePath"):
        debug_log(
            "analyze-face-check-before",
            uploadId=request.get("uploadId"),
            sourcePath=request.get("sourcePath"),
        )
        face = detect_primary_face(image, request)
        debug_log(
            "analyze-face-check-after",
            uploadId=request.get("uploadId"),
            accepted=True,
            selectedFace=face["box"],
            rawFaces=face.get("debug", {}).get("rawFaces", []),
            rawEyes=face.get("debug", {}).get("rawEyes", []),
        )
    metrics = image_metrics(image)
    return {
        "score": metrics["score"],
        "summary": summarize(metrics["metrics"]),
        "metrics": metrics["metrics"],
    }


def handle_preview(request):
    preview_started = now_ms()
    pipeline = run_face_aware_pipeline(request, add_preview_watermark=True)
    processed = pipeline["processed"]
    used_gpu = pipeline["usedGpu"]
    identity_context = pipeline["identityContext"]
    identity_meta = pipeline["identityMeta"]
    rotated_to_portrait = pipeline["rotatedToPortrait"]

    output_path = Path(request["outputPath"])
    timed_call("preview-output-dir-create", lambda: output_path.parent.mkdir(parents=True, exist_ok=True), outputPath=str(output_path))
    timed_call("preview-output-save", lambda: processed.save(output_path, format="PNG"), outputPath=str(output_path))
    debug_log(
        "preview-handle-complete",
        uploadId=request.get("uploadId"),
        totalDurationMs=now_ms() - preview_started,
        usedGpu=used_gpu,
        rotatedToPortrait=rotated_to_portrait,
        identityGenerationUsed=identity_meta["identityGenerationUsed"],
        identityGenerationMode=identity_meta["identityGenerationMode"],
    )

    return {
        "preset": request.get("preset", "natural"),
        "previewPath": str(output_path.resolve()),
        "watermarkText": request.get("watermarkText", "RizzUp Preview"),
        "usedGpu": used_gpu,
        "identityGenerationUsed": identity_meta["identityGenerationUsed"],
        "identityGenerationMode": identity_meta["identityGenerationMode"],
        "identityFallbackReason": identity_meta["identityFallbackReason"],
        "rotatedToPortrait": rotated_to_portrait,
        "rotationDegrees": pipeline["rotationDegrees"],
        "width": int(processed.width),
        "height": int(processed.height),
        "identityContext": {
            "embeddingSize": len(identity_context["embedding"]),
            "faceBox": identity_context["faceBox"],
        },
    }


def handle_validate_upload(request):
    image = timed_call(
        "source-open",
        lambda: open_source_image(request.get("sourcePath")),
        uploadId=request.get("uploadId"),
        sourcePath=request.get("sourcePath"),
    )
    image, rotated_to_portrait, face, rotation_degrees = timed_call(
        "portrait-normalization",
        lambda: normalize_to_best_portrait(image, request),
        uploadId=request.get("uploadId"),
    )
    return {
        "faceDetection": {
            "box": face["box"],
            "landmarks": {
                key: list(value) for key, value in face["landmarks"].items()
            },
            "debug": face.get("debug", {"rawFaces": [], "rawEyes": []}),
            "rotatedToPortrait": rotated_to_portrait,
            "rotationDegrees": rotation_degrees,
        }
    }


def handle_final(request):
    final_started = now_ms()
    pipeline = run_face_aware_pipeline(request, add_preview_watermark=False)
    processed = pipeline["processed"]
    used_gpu = pipeline["usedGpu"]
    identity_meta = pipeline["identityMeta"]
    rotated_to_portrait = pipeline["rotatedToPortrait"]
    processed = processed.resize((processed.width * 2, processed.height * 2), Image.Resampling.LANCZOS)

    output_path = Path(request["outputPath"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(output_path, format="PNG")

    debug_log(
        "final-handle-complete",
        uploadId=request.get("uploadId"),
        totalDurationMs=now_ms() - final_started,
        usedGpu=used_gpu,
        rotatedToPortrait=rotated_to_portrait,
        identityGenerationUsed=identity_meta["identityGenerationUsed"],
        identityGenerationMode=identity_meta["identityGenerationMode"],
    )

    return {
        "preset": request.get("preset", "natural"),
        "finalImagePath": str(output_path.resolve()),
        "usedGpu": used_gpu,
        "identityGenerationUsed": identity_meta["identityGenerationUsed"],
        "identityGenerationMode": identity_meta["identityGenerationMode"],
        "identityFallbackReason": identity_meta["identityFallbackReason"],
        "rotatedToPortrait": rotated_to_portrait,
        "rotationDegrees": pipeline["rotationDegrees"],
        "width": int(processed.width),
        "height": int(processed.height),
    }


def main():
    request = load_request()
    action = request.get("action")
    if action == "analyze":
        result = handle_analyze(request)
    elif action == "validate_upload":
        result = handle_validate_upload(request)
    elif action == "preview":
        result = handle_preview(request)
    elif action == "final":
        result = handle_final(request)
    else:
        raise ValueError(f"Unsupported action: {action}")

    sys.stdout.write(json.dumps(sanitize_for_json(result)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if isinstance(exc, PipelineValidationError):
            debug_log("preview-face-check-after", accepted=False, errorCode=exc.code, errorMessage=exc.message)
            sys.stderr.write(json.dumps(sanitize_for_json(exc.to_dict())))
        else:
            sys.stderr.write(str(exc))
        sys.exit(1)
