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

function optionalPathEnv(name: string, fallback: string): string {
  const raw = process.env[name]?.trim();
  return path.resolve(process.cwd(), raw || fallback);
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): WorkerConfig {
  const cwd = process.cwd();

  return {
    netlifySiteId: requireEnv("NETLIFY_SITE_ID"),
    netlifyAccessToken: requireEnv("NETLIFY_ACCESS_TOKEN"),
    queueStore: env.RIZZUP_QUEUE_STORE?.trim() || "rizzup-job-queue",
    statusStore: env.RIZZUP_STATUS_STORE?.trim() || "rizzup-job-status",
    resultsStore: env.RIZZUP_RESULTS_STORE?.trim() || "rizzup-job-results",
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
    resultsDir: optionalPathEnv("RIZZUP_RESULTS_DIR", "artifacts"),
    sourceImageRoot: env.RIZZUP_SOURCE_IMAGE_ROOT?.trim()
      ? path.resolve(cwd, env.RIZZUP_SOURCE_IMAGE_ROOT.trim())
      : undefined,
    resultsPublicBaseUrl: env.RIZZUP_RESULTS_PUBLIC_BASE_URL?.trim() || undefined,
    pythonExecutable: optionalPathEnv("RIZZUP_PYTHON_EXECUTABLE", ".venv\\Scripts\\python.exe"),
    pythonScript: optionalPathEnv("RIZZUP_PYTHON_SCRIPT", "scripts\\gpu_pipeline.py")
  };
}
