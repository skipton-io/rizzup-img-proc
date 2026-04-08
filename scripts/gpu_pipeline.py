import json
import sys
from pathlib import Path

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


def load_request():
    return json.loads(sys.stdin.read())


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
    metrics = image_metrics(image)
    return {
        "score": metrics["score"],
        "summary": summarize(metrics["metrics"]),
        "metrics": metrics["metrics"],
    }


def handle_preview(request):
    image = open_or_placeholder(request.get("sourcePath"))
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    processed, used_gpu = apply_preset_gpu(image, request.get("preset", "natural"))
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
        sys.stderr.write(str(exc))
        sys.exit(1)
