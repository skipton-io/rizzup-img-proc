import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


PRESET_TUNING = {
    "natural": {"brightness": 1.03, "contrast": 1.05, "saturation": 1.02, "temperature": 0.0},
    "professional": {"brightness": 1.02, "contrast": 1.08, "saturation": 0.92, "temperature": -0.03},
    "lifestyle": {"brightness": 1.06, "contrast": 1.10, "saturation": 1.14, "temperature": 0.04},
    "fitness": {"brightness": 1.04, "contrast": 1.18, "saturation": 1.10, "temperature": 0.02},
    "travel": {"brightness": 1.08, "contrast": 1.12, "saturation": 1.18, "temperature": 0.05},
}

FACE_NOT_DETECTED_MESSAGE = "No face detected. Please upload a clear photo with one visible face."


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
    payload = {"event": event, **details}
    sys.stderr.write(f"{json.dumps(payload)}\n")
    sys.stderr.flush()


def open_or_placeholder(source_path):
    if source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return Image.open(candidate).convert("RGB")

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

    return Image.open(candidate).convert("RGB")


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
        left_eye = (x + eyes[0][0] + eyes[0][2] // 2, y + eyes[0][1] + eyes[0][3] // 2)
        right_eye = (x + eyes[1][0] + eyes[1][2] // 2, y + eyes[1][1] + eyes[1][3] // 2)
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


def add_watermark(image, watermark_text):
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
    image = open_source_image(request.get("sourcePath"))
    debug_log(
        "preview-face-check-before",
        uploadId=request.get("uploadId"),
        preset=request.get("preset", "natural"),
        sourcePath=request.get("sourcePath"),
    )
    face = detect_primary_face(image, request)
    debug_log(
        "preview-face-check-after",
        uploadId=request.get("uploadId"),
        accepted=True,
        selectedFace=face["box"],
        rawFaces=face.get("debug", {}).get("rawFaces", []),
        rawEyes=face.get("debug", {}).get("rawEyes", []),
    )
    identity_context = build_identity_context(image, face)
    processed = correct_lighting(image)
    processed = subtle_skin_cleanup(processed, face)
    processed = improve_background(processed, face)
    processed = optimize_framing(processed, face)
    processed, used_gpu = apply_preset_gpu(processed, request.get("preset", "natural"))
    processed = add_watermark(processed, request.get("watermarkText", "RizzUp Preview"))

    output_path = Path(request["outputPath"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(output_path, format="PNG")

    return {
        "preset": request.get("preset", "natural"),
        "previewPath": str(output_path.resolve()),
        "watermarkText": request.get("watermarkText", "RizzUp Preview"),
        "usedGpu": used_gpu,
        "width": int(processed.width),
        "height": int(processed.height),
        "identityContext": {
            "embeddingSize": len(identity_context["embedding"]),
            "faceBox": identity_context["faceBox"],
        },
    }


def handle_final(request):
    image = open_or_placeholder(request.get("sourcePath"))
    processed, used_gpu = apply_preset_gpu(image, request.get("preset", "natural"))
    processed = processed.resize((processed.width * 2, processed.height * 2), Image.Resampling.LANCZOS)

    output_path = Path(request["outputPath"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(output_path, format="PNG")

    return {
        "preset": request.get("preset", "natural"),
        "finalImagePath": str(output_path.resolve()),
        "usedGpu": used_gpu,
        "width": int(processed.width),
        "height": int(processed.height),
    }


def main():
    request = load_request()
    action = request.get("action")
    if action == "analyze":
        result = handle_analyze(request)
    elif action == "preview":
        result = handle_preview(request)
    elif action == "final":
        result = handle_final(request)
    else:
        raise ValueError(f"Unsupported action: {action}")

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if isinstance(exc, PipelineValidationError):
            debug_log("preview-face-check-after", accepted=False, errorCode=exc.code, errorMessage=exc.message)
            sys.stderr.write(json.dumps(exc.to_dict()))
        else:
            sys.stderr.write(str(exc))
        sys.exit(1)
