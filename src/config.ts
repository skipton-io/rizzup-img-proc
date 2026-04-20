import crypto from "node:crypto";
import path from "node:path";
import { WorkerConfig } from "./types";

function requireEnv(name: string, env: NodeJS.ProcessEnv = process.env): string {
  const value = env[name];
  if (!value || !value.trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value.trim();
}

function numberEnv(name: string, fallback: number, env: NodeJS.ProcessEnv = process.env): number {
  const raw = env[name];
  if (!raw) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Environment variable ${name} must be a positive number`);
  }

  return parsed;
}

function optionalPathEnv(name: string, fallback: string, env: NodeJS.ProcessEnv = process.env): string {
  const raw = env[name]?.trim();
  return path.resolve(process.cwd(), raw || fallback);
}

function optionalResolvedPath(name: string, env: NodeJS.ProcessEnv = process.env): string | undefined {
  const raw = env[name]?.trim();
  return raw ? path.resolve(process.cwd(), raw) : undefined;
}

function booleanEnv(name: string, fallback: boolean, env: NodeJS.ProcessEnv = process.env): boolean {
  const raw = env[name]?.trim().toLowerCase();
  if (!raw) return fallback;
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  throw new Error(`Environment variable ${name} must be a boolean`);
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): WorkerConfig {
  const cwd = process.cwd();
  const accessToken = env.NETLIFY_ACCESS_TOKEN?.trim() || env.NETLIFY_AUTH_TOKEN?.trim();
  const archiveBackend = env.RIZZUP_ARCHIVE_BACKEND?.trim().toLowerCase() === "sftp" ? "sftp" : "local";
  const pusherAppId = env.PUSHER_APP_ID?.trim() || "";
  const pusherKey = env.PUSHER_KEY?.trim() || "";
  const pusherSecret = env.PUSHER_SECRET?.trim() || "";
  const pusherCluster = env.PUSHER_CLUSTER?.trim() || "";
  const resultsDir = optionalPathEnv("RIZZUP_RESULTS_DIR", "artifacts", env);
  const sourceImageRoot = env.RIZZUP_SOURCE_IMAGE_ROOT?.trim()
    ? path.resolve(cwd, env.RIZZUP_SOURCE_IMAGE_ROOT.trim())
    : archiveBackend === "sftp"
      ? path.resolve(resultsDir, "source-images")
      : undefined;

  if (!accessToken) {
    throw new Error("Missing required environment variable: NETLIFY_ACCESS_TOKEN");
  }

  return {
    netlifySiteId: requireEnv("NETLIFY_SITE_ID", env),
    netlifyAccessToken: accessToken,
    queueStore: env.RIZZUP_QUEUE_STORE?.trim() || "rizzup-job-queue",
    statusStore: env.RIZZUP_STATUS_STORE?.trim() || "rizzup-job-status",
    resultsStore: env.RIZZUP_RESULTS_STORE?.trim() || "rizzup-job-results",
    assetsStore: env.RIZZUP_ASSETS_STORE?.trim() || "rizzup-job-assets",
    locksStore: env.RIZZUP_LOCKS_STORE?.trim() || "rizzup-job-locks",
    deadLetterStore: env.RIZZUP_DEAD_LETTER_STORE?.trim() || "rizzup-job-dead-letter",
    maxRuntimeMs: numberEnv("RIZZUP_MAX_RUNTIME_MS", 55_000, env),
    pollIntervalMs: numberEnv("RIZZUP_POLL_INTERVAL_MS", 5_000, env),
    maxJobsPerPoll: numberEnv("RIZZUP_MAX_JOBS_PER_POLL", 1, env),
    lockTtlMs: numberEnv("RIZZUP_LOCK_TTL_MS", 1_800_000, env),
    maxAttempts: numberEnv("RIZZUP_MAX_ATTEMPTS", 3, env),
    retryBaseDelayMs: numberEnv("RIZZUP_RETRY_BASE_DELAY_MS", 5_000, env),
    retryMaxDelayMs: numberEnv("RIZZUP_RETRY_MAX_DELAY_MS", 60_000, env),
    workerId: env.RIZZUP_WORKER_ID?.trim() || `worker_${crypto.randomUUID().slice(0, 8)}`,
    previewWatermarkText: env.RIZZUP_PREVIEW_WATERMARK_TEXT?.trim() || "RizzUp Preview",
    previewWatermarkLogoPath: optionalResolvedPath(
      "RIZZUP_PREVIEW_WATERMARK_LOGO_PATH",
      env
    ) || path.resolve(cwd, "..", "rizzup.co.uk", "public", "brand", "rizzup-logo.png"),
    resultsDir,
    archiveBackend,
    imageArchiveRoot:
      env.RIZZUP_IMAGE_ARCHIVE_ROOT?.trim() ||
      "\\\\CODO-DIGITAL-L\\web\\rizzup.co.uk\\image-jobs",
    sourceImageRoot,
    localRenderRoot: path.resolve(resultsDir, "renders"),
    sftpHost: env.RIZZUP_SFTP_HOST?.trim() || undefined,
    sftpPort: numberEnv("RIZZUP_SFTP_PORT", 22, env),
    sftpUsername: env.RIZZUP_SFTP_USERNAME?.trim() || undefined,
    sftpPassword: env.RIZZUP_SFTP_PASSWORD?.trim() || undefined,
    sftpStrictHostKey: booleanEnv("RIZZUP_SFTP_STRICT_HOST_KEY", false, env),
    sftpHostKey: env.RIZZUP_SFTP_HOST_KEY?.trim() || undefined,
    resultsPublicBaseUrl: env.RIZZUP_RESULTS_PUBLIC_BASE_URL?.trim() || undefined,
    pusher:
      pusherAppId && pusherKey && pusherSecret && pusherCluster
        ? {
            appId: pusherAppId,
            key: pusherKey,
            secret: pusherSecret,
            cluster: pusherCluster
          }
        : undefined,
    pythonExecutable: optionalPathEnv("RIZZUP_PYTHON_EXECUTABLE", ".venv\\Scripts\\python.exe", env),
    pythonScript: optionalPathEnv("RIZZUP_PYTHON_SCRIPT", "scripts\\gpu_pipeline.py", env),
    faceCascadePath: optionalResolvedPath("RIZZUP_FACE_CASCADE_PATH", env),
    eyeCascadePath: optionalResolvedPath("RIZZUP_EYE_CASCADE_PATH", env),
    fireRedEnabled: booleanEnv("RIZZUP_FIRERED_ENABLED", true, env),
    fireRedModelId: env.RIZZUP_FIRERED_MODEL_ID?.trim() || "FireRedTeam/FireRed-Image-Edit-1.1",
    fireRedLoraRepo:
      env.RIZZUP_FIRERED_LORA_REPO?.trim() || "FireRedTeam/FireRed-Image-Edit-LoRA-Zoo",
    fireRedLoraWeight:
      env.RIZZUP_FIRERED_LORA_WEIGHT?.trim() || "FireRed-Image-Edit-Makeup.safetensors",
    fireRedLoraAdapterName: env.RIZZUP_FIRERED_LORA_ADAPTER_NAME?.trim() || "makeup",
    fireRedPrompt: env.RIZZUP_FIRERED_PROMPT?.trim() || "Western makeup",
    fireRedInferenceSteps: numberEnv("RIZZUP_FIRERED_INFERENCE_STEPS", 30, env),
    fireRedTrueCfgScale: numberEnv("RIZZUP_FIRERED_TRUE_CFG_SCALE", 4, env),
    analysisMaxSize: numberEnv("RIZZUP_ANALYSIS_MAX_SIZE", 100, env),
    previewMaxSize: numberEnv("RIZZUP_PREVIEW_MAX_SIZE", 512, env),
    finalDecisionMaxSize: numberEnv("RIZZUP_FINAL_DECISION_MAX_SIZE", 512, env),
    finalMinWidth: numberEnv("RIZZUP_FINAL_MIN_WIDTH", 1024, env),
    finalMinHeight: numberEnv("RIZZUP_FINAL_MIN_HEIGHT", 1280, env)
  };
}
