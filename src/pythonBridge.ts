import { spawn } from "node:child_process";
import path from "node:path";
import {
  CachedFaceDetection,
  HandlerContext,
  PhotoQualityResult,
  PreviewResult,
  UploadPhotoResult
} from "./types";

type PythonAnalyzeResponse = Omit<PhotoQualityResult, "uploadId" | "analyzedAt">;
type PythonValidateUploadResponse = {
  faceDetection: CachedFaceDetection;
};
type PythonPreviewResponse = Omit<PreviewResult, "uploadId" | "generatedAt" | "previewAssetId"> & {
  previewPath: string;
};
type PythonFinalResponse = Omit<
  import("./types").FinalImageResult,
  "unlockId" | "checkoutSessionId" | "uploadId" | "generatedAt" | "finalImageAssetId" | "plan"
> & { finalImagePath: string };

type PythonRequest =
  | {
      action: "validate_upload";
      uploadId: string;
      sourcePath?: string | null;
      faceCascadePath?: string | null;
      eyeCascadePath?: string | null;
    }
  | {
      action: "analyze";
      uploadId: string;
      sourcePath?: string | null;
      width?: number | null;
      height?: number | null;
      analysisMaxSize: number;
      faceDetection?: CachedFaceDetection | null;
    }
  | {
      action: "preview";
      uploadId: string;
      preset: string;
      sourcePath?: string | null;
      outputPath: string;
      watermarkText: string;
      watermarkLogoPath?: string | null;
      faceDetection?: CachedFaceDetection | null;
      faceCascadePath?: string | null;
      eyeCascadePath?: string | null;
      fireRedEnabled: boolean;
      fireRedModelId: string;
      fireRedPrompt: string;
      fireRedInferenceSteps: number;
      fireRedTrueCfgScale: number;
      previewMaxSize: number;
    }
  | {
      action: "final";
      uploadId: string;
      preset: string;
      sourcePath?: string | null;
      outputPath: string;
      faceDetection?: CachedFaceDetection | null;
      faceCascadePath?: string | null;
      eyeCascadePath?: string | null;
      fireRedEnabled: boolean;
      fireRedModelId: string;
      fireRedPrompt: string;
      fireRedInferenceSteps: number;
      fireRedTrueCfgScale: number;
      finalDecisionMaxSize: number;
      finalMinWidth: number;
      finalMinHeight: number;
    };

function resolveSourcePath(
  upload: UploadPhotoResult | null,
  context: HandlerContext
): string | null {
  const sourcePath = upload?.sourcePath?.trim();
  if (!sourcePath) return null;
  if (path.isAbsolute(sourcePath)) {
    return sourcePath;
  }

  if (!context.config.sourceImageRoot) {
    return path.resolve(process.cwd(), sourcePath);
  }

  return path.resolve(context.config.sourceImageRoot, sourcePath);
}

type StructuredPythonError = {
  code?: string;
  message?: string;
  retryable?: boolean;
  details?: Record<string, unknown>;
};

export class PipelineJobError extends Error {
  public readonly code?: string;
  public readonly retryable: boolean;
  public readonly details?: Record<string, unknown>;

  constructor(message: string, options?: StructuredPythonError) {
    super(message);
    this.name = "PipelineJobError";
    this.code = options?.code;
    this.retryable = options?.retryable ?? true;
    this.details = options?.details;
  }
}

function parseStructuredError(stderr: string): PipelineJobError | null {
  const lines = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) return null;

  try {
    const parsed = JSON.parse(lines[lines.length - 1]) as StructuredPythonError;
    if (!parsed?.message) return null;
    return new PipelineJobError(parsed.message, parsed);
  } catch {
    return null;
  }
}

function logPythonBridge(event: string, details: Record<string, unknown>): void {
  const formatted = Object.entries(details)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(" ");
  process.stdout.write(`[rizzup-python-bridge] ${event}${formatted ? ` ${formatted}` : ""}\n`);
}

function parseStructuredSuccess<T>(stdout: string): T {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new Error("Python pipeline returned no stdout.");
  }

  try {
    return JSON.parse(trimmed) as T;
  } catch {
    const lines = trimmed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    for (let index = lines.length - 1; index >= 0; index -= 1) {
      try {
        return JSON.parse(lines[index]) as T;
      } catch {
        continue;
      }
    }
  }

  throw new Error(`Could not parse python output: ${trimmed.slice(0, 200)}`);
}

async function runPython<T>(request: PythonRequest, context: HandlerContext): Promise<T> {
  return await new Promise<T>((resolve, reject) => {
    const startedAt = Date.now();
    logPythonBridge("spawn-start", {
      action: request.action,
      uploadId: "uploadId" in request ? request.uploadId : undefined,
      pythonExecutable: context.config.pythonExecutable,
      pythonScript: context.config.pythonScript
    });
    const child = spawn(context.config.pythonExecutable, [context.config.pythonScript], {
      cwd: process.cwd(),
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      logPythonBridge("spawn-error", {
        action: request.action,
        uploadId: "uploadId" in request ? request.uploadId : undefined,
        durationMs: Date.now() - startedAt,
        error: error.message
      });
      reject(error);
    });

    child.on("close", (code) => {
      const durationMs = Date.now() - startedAt;
      logPythonBridge("spawn-close", {
        action: request.action,
        uploadId: "uploadId" in request ? request.uploadId : undefined,
        code,
        durationMs,
        stdoutBytes: stdout.length,
        stderrBytes: stderr.length
      });
      if (code !== 0) {
        const structured = parseStructuredError(stderr);
        if (structured) {
          if (structured.details) {
            process.stdout.write(
              `[rizzup-python-debug] error-details ${JSON.stringify(structured.details)}\n`
            );
          }
          reject(structured);
          return;
        }

        reject(new Error(`Python pipeline exited with code ${code}. ${stderr.trim() || "No stderr output"}`));
        return;
      }

      if (stderr.trim()) {
        for (const line of stderr.trim().split(/\r?\n/)) {
          process.stdout.write(`[rizzup-python-debug] ${line}\n`);
        }
      }

      try {
        resolve(parseStructuredSuccess<T>(stdout));
      } catch (error) {
        reject(error as Error);
      }
    });

    logPythonBridge("stdin-write", {
      action: request.action,
      uploadId: "uploadId" in request ? request.uploadId : undefined,
      requestBytes: JSON.stringify(request).length
    });
    child.stdin.write(JSON.stringify(request));
    child.stdin.end();
  });
}

export async function analyzeWithPython(
  uploadId: string,
  upload: UploadPhotoResult | null,
  context: HandlerContext
): Promise<PhotoQualityResult> {
  const response = await runPython<PythonAnalyzeResponse>(
    {
      action: "analyze",
      uploadId,
      sourcePath: resolveSourcePath(upload, context),
      width: upload?.width ?? null,
      height: upload?.height ?? null,
      analysisMaxSize: context.config.analysisMaxSize,
      faceDetection: upload?.faceDetection ?? null
    },
    context
  );

  return {
    uploadId,
    ...response,
    analyzedAt: new Date().toISOString()
  };
}

export async function validateUploadWithPython(
  uploadId: string,
  upload: UploadPhotoResult | null,
  context: HandlerContext
): Promise<CachedFaceDetection> {
  const response = await runPython<PythonValidateUploadResponse>(
    {
      action: "validate_upload",
      uploadId,
      sourcePath: resolveSourcePath(upload, context),
      faceCascadePath: context.config.faceCascadePath ?? null,
      eyeCascadePath: context.config.eyeCascadePath ?? null
    },
    context
  );

  return response.faceDetection;
}

export async function generatePreviewWithPython(
  uploadId: string,
  preset: string,
  outputPath: string,
  upload: UploadPhotoResult | null,
  context: HandlerContext
): Promise<PreviewResult> {
  const response = await runPython<PythonPreviewResponse>(
    {
      action: "preview",
      uploadId,
      preset,
      sourcePath: resolveSourcePath(upload, context),
      outputPath,
      watermarkText: context.config.previewWatermarkText,
      watermarkLogoPath: context.config.previewWatermarkLogoPath ?? null,
      faceDetection: upload?.faceDetection ?? null,
      faceCascadePath: context.config.faceCascadePath ?? null,
      eyeCascadePath: context.config.eyeCascadePath ?? null,
      fireRedEnabled: context.config.fireRedEnabled,
      fireRedModelId: context.config.fireRedModelId,
      fireRedPrompt: context.config.fireRedPrompt,
      fireRedInferenceSteps: context.config.fireRedInferenceSteps,
      fireRedTrueCfgScale: context.config.fireRedTrueCfgScale,
      previewMaxSize: context.config.previewMaxSize
    },
    context
  );
  const { previewPath: _previewPath, ...rest } = response;

  return {
    uploadId,
    ...rest,
    previewPath: outputPath,
    previewAssetId: "",
    generatedAt: new Date().toISOString()
  };
}

export async function generateFinalImageWithPython(
  unlockId: string,
  checkoutSessionId: string,
  uploadId: string,
  preset: string,
  outputPath: string,
  plan: string,
  upload: UploadPhotoResult | null,
  context: HandlerContext
): Promise<import("./types").FinalImageResult> {
  const response = await runPython<PythonFinalResponse>(
    {
      action: "final",
      uploadId,
      preset,
      sourcePath: resolveSourcePath(upload, context),
      outputPath,
      faceDetection: upload?.faceDetection ?? null,
      faceCascadePath: context.config.faceCascadePath ?? null,
      eyeCascadePath: context.config.eyeCascadePath ?? null,
      fireRedEnabled: context.config.fireRedEnabled,
      fireRedModelId: context.config.fireRedModelId,
      fireRedPrompt: context.config.fireRedPrompt,
      fireRedInferenceSteps: context.config.fireRedInferenceSteps,
      fireRedTrueCfgScale: context.config.fireRedTrueCfgScale,
      finalDecisionMaxSize: context.config.finalDecisionMaxSize,
      finalMinWidth: context.config.finalMinWidth,
      finalMinHeight: context.config.finalMinHeight
    },
    context
  );
  const { finalImagePath: _finalImagePath, ...rest } = response;

  return {
    unlockId,
    checkoutSessionId,
    uploadId,
    plan,
    ...rest,
    finalImagePath: outputPath,
    finalImageAssetId: "",
    generatedAt: new Date().toISOString()
  };
}
