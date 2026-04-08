import crypto from "node:crypto";
import path from "node:path";
import { WorkerConfig } from "./types";

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value || !value.trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value.trim();
}

function numberEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Environment variable ${name} must be a positive number`);
  }

  return parsed;
}

function booleanEnv(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (!raw) return fallback;
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  throw new Error(`Environment variable ${name} must be a boolean value`);
}

function optionalPathEnv(name: string, fallback: string): string {
  const raw = process.env[name]?.trim();
  return path.resolve(process.cwd(), raw || fallback);
}

function optionalResolvedPath(name: string, env: NodeJS.ProcessEnv = process.env): string | undefined {
  const raw = env[name]?.trim();
  return raw ? path.resolve(process.cwd(), raw) : undefined;
}

function previewIdentityFallbackModeEnv(
  name: string,
  fallback: "heuristic" | "error"
): "heuristic" | "error" {
  const raw = process.env[name]?.trim().toLowerCase();
  if (!raw) return fallback;
  if (raw === "heuristic" || raw === "error") return raw;
  throw new Error(`Environment variable ${name} must be either "heuristic" or "error"`);
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): WorkerConfig {
  const cwd = process.cwd();
  const accessToken = env.NETLIFY_ACCESS_TOKEN?.trim() || env.NETLIFY_AUTH_TOKEN?.trim();

  if (!accessToken) {
    throw new Error("Missing required environment variable: NETLIFY_ACCESS_TOKEN");
  }

  return {
    netlifySiteId: requireEnv("NETLIFY_SITE_ID"),
    netlifyAccessToken: accessToken,
    queueStore: env.RIZZUP_QUEUE_STORE?.trim() || "rizzup-job-queue",
    statusStore: env.RIZZUP_STATUS_STORE?.trim() || "rizzup-job-status",
    resultsStore: env.RIZZUP_RESULTS_STORE?.trim() || "rizzup-job-results",
    assetsStore: env.RIZZUP_ASSETS_STORE?.trim() || "rizzup-job-assets",
    locksStore: env.RIZZUP_LOCKS_STORE?.trim() || "rizzup-job-locks",
    deadLetterStore: env.RIZZUP_DEAD_LETTER_STORE?.trim() || "rizzup-job-dead-letter",
    pollIntervalMs: numberEnv("RIZZUP_POLL_INTERVAL_MS", 5_000),
    maxJobsPerPoll: numberEnv("RIZZUP_MAX_JOBS_PER_POLL", 8),
    lockTtlMs: numberEnv("RIZZUP_LOCK_TTL_MS", 120_000),
    maxAttempts: numberEnv("RIZZUP_MAX_ATTEMPTS", 3),
    retryBaseDelayMs: numberEnv("RIZZUP_RETRY_BASE_DELAY_MS", 5_000),
    retryMaxDelayMs: numberEnv("RIZZUP_RETRY_MAX_DELAY_MS", 60_000),
    workerId: env.RIZZUP_WORKER_ID?.trim() || `worker_${crypto.randomUUID().slice(0, 8)}`,
    previewWatermarkText: env.RIZZUP_PREVIEW_WATERMARK_TEXT?.trim() || "RizzUp Preview",
    previewWatermarkLogoPath: optionalResolvedPath(
      "RIZZUP_PREVIEW_WATERMARK_LOGO_PATH",
      env
    ) || path.resolve(cwd, "..", "rizzup.co.uk", "public", "brand", "rizzup-logo.png"),
    resultsDir: optionalPathEnv("RIZZUP_RESULTS_DIR", "artifacts"),
    imageArchiveRoot:
      env.RIZZUP_IMAGE_ARCHIVE_ROOT?.trim() ||
      "\\\\CODO-DIGITAL-L\\web\\rizzup.co.uk\\image-jobs",
    sourceImageRoot: env.RIZZUP_SOURCE_IMAGE_ROOT?.trim()
      ? path.resolve(cwd, env.RIZZUP_SOURCE_IMAGE_ROOT.trim())
      : undefined,
    resultsPublicBaseUrl: env.RIZZUP_RESULTS_PUBLIC_BASE_URL?.trim() || undefined,
    pythonExecutable: optionalPathEnv("RIZZUP_PYTHON_EXECUTABLE", ".venv\\Scripts\\python.exe"),
    pythonScript: optionalPathEnv("RIZZUP_PYTHON_SCRIPT", "scripts\\gpu_pipeline.py"),
    faceCascadePath: optionalResolvedPath("RIZZUP_FACE_CASCADE_PATH", env),
    eyeCascadePath: optionalResolvedPath("RIZZUP_EYE_CASCADE_PATH", env),
    previewIdentityEnabled: booleanEnv("RIZZUP_PREVIEW_IDENTITY_ENABLED", true),
    previewIdentityFallbackMode: previewIdentityFallbackModeEnv(
      "RIZZUP_PREVIEW_IDENTITY_FALLBACK_MODE",
      "heuristic"
    ),
    previewIdentityCacheDir: optionalPathEnv("RIZZUP_PREVIEW_IDENTITY_CACHE_DIR", ".cache\\instantid"),
    previewIdentityPipelinePath: optionalPathEnv(
      "RIZZUP_PREVIEW_IDENTITY_PIPELINE_PATH",
      "third_party\\InstantID\\pipeline_stable_diffusion_xl_instantid.py"
    ),
    previewIdentityCheckpointDir: optionalPathEnv(
      "RIZZUP_PREVIEW_IDENTITY_CHECKPOINT_DIR",
      "third_party\\InstantID\\checkpoints"
    ),
    previewIdentityFaceEncoderRoot: optionalPathEnv(
      "RIZZUP_PREVIEW_IDENTITY_FACE_ENCODER_ROOT",
      "third_party\\InstantID\\models"
    ),
    previewIdentityBaseModel:
      env.RIZZUP_PREVIEW_IDENTITY_BASE_MODEL?.trim() || "stabilityai/stable-diffusion-xl-base-1.0",
    previewIdentityPromptTemplate: env.RIZZUP_PREVIEW_IDENTITY_PROMPT_TEMPLATE?.trim() || undefined,
    previewIdentityNegativePrompt:
      env.RIZZUP_PREVIEW_IDENTITY_NEGATIVE_PROMPT?.trim() ||
      "low quality, blurry, deformed, distorted face, extra limbs, duplicate features, waxy skin, oversmoothed skin, uncanny expression",
    previewIdentitySteps: numberEnv("RIZZUP_PREVIEW_IDENTITY_STEPS", 30),
    previewIdentityGuidanceScale: numberEnv("RIZZUP_PREVIEW_IDENTITY_GUIDANCE_SCALE", 4.5),
    previewIdentityControlScale: numberEnv("RIZZUP_PREVIEW_IDENTITY_CONTROL_SCALE", 0.72),
    previewIdentityAdapterScale: numberEnv("RIZZUP_PREVIEW_IDENTITY_ADAPTER_SCALE", 0.68),
    previewIdentityBlendStrength: numberEnv("RIZZUP_PREVIEW_IDENTITY_BLEND_STRENGTH", 0.35)
  };
}
