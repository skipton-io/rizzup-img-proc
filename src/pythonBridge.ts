import { spawn } from "node:child_process";
import path from "node:path";
import { HandlerContext, PhotoQualityResult, PreviewResult, UploadPhotoResult } from "./types";

type PythonAnalyzeResponse = Omit<PhotoQualityResult, "uploadId" | "analyzedAt">;
type PythonPreviewResponse = Omit<PreviewResult, "uploadId" | "generatedAt" | "previewAssetId"> & {
  previewPath: string;
};
type PythonFinalResponse = Omit<
  import("./types").FinalImageResult,
  "unlockId" | "checkoutSessionId" | "uploadId" | "generatedAt" | "finalImageAssetId" | "plan"
> & { finalImagePath: string };

type PythonRequest =
  | {
      action: "analyze";
      uploadId: string;
      sourcePath?: string | null;
      width?: number | null;
      height?: number | null;
    }
  | {
      action: "preview";
      uploadId: string;
      preset: string;
      sourcePath?: string | null;
      outputPath: string;
      watermarkText: string;
      faceCascadePath?: string | null;
      eyeCascadePath?: string | null;
      previewIdentityEnabled: boolean;
      previewIdentityFallbackMode: "heuristic" | "error";
      previewIdentityCacheDir: string;
      previewIdentityPipelinePath: string;
      previewIdentityCheckpointDir: string;
      previewIdentityFaceEncoderRoot: string;
      previewIdentityBaseModel: string;
      previewIdentityPromptTemplate?: string | null;
      previewIdentityNegativePrompt: string;
      previewIdentitySteps: number;
      previewIdentityGuidanceScale: number;
      previewIdentityControlScale: number;
      previewIdentityAdapterScale: number;
      previewIdentityBlendStrength: number;
    }
  | {
      action: "final";
      uploadId: string;
      preset: string;
      sourcePath?: string | null;
      outputPath: string;
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
        resolve(JSON.parse(stdout) as T);
      } catch (error) {
        reject(new Error(`Could not parse python output: ${(error as Error).message}`));
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
      height: upload?.height ?? null
    },
    context
  );

  return {
    uploadId,
    ...response,
    analyzedAt: new Date().toISOString()
  };
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
      faceCascadePath: context.config.faceCascadePath ?? null,
      eyeCascadePath: context.config.eyeCascadePath ?? null,
      previewIdentityEnabled: context.config.previewIdentityEnabled,
      previewIdentityFallbackMode: context.config.previewIdentityFallbackMode,
      previewIdentityCacheDir: context.config.previewIdentityCacheDir,
      previewIdentityPipelinePath: context.config.previewIdentityPipelinePath,
      previewIdentityCheckpointDir: context.config.previewIdentityCheckpointDir,
      previewIdentityFaceEncoderRoot: context.config.previewIdentityFaceEncoderRoot,
      previewIdentityBaseModel: context.config.previewIdentityBaseModel,
      previewIdentityPromptTemplate: context.config.previewIdentityPromptTemplate ?? null,
      previewIdentityNegativePrompt: context.config.previewIdentityNegativePrompt,
      previewIdentitySteps: context.config.previewIdentitySteps,
      previewIdentityGuidanceScale: context.config.previewIdentityGuidanceScale,
      previewIdentityControlScale: context.config.previewIdentityControlScale,
      previewIdentityAdapterScale: context.config.previewIdentityAdapterScale,
      previewIdentityBlendStrength: context.config.previewIdentityBlendStrength
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
      outputPath
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
