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

async function runPython<T>(request: PythonRequest, context: HandlerContext): Promise<T> {
  return await new Promise<T>((resolve, reject) => {
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
      reject(error);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `Python pipeline exited with code ${code}. ${stderr.trim() || "No stderr output"}`
          )
        );
        return;
      }

      try {
        resolve(JSON.parse(stdout) as T);
      } catch (error) {
        reject(new Error(`Could not parse python output: ${(error as Error).message}`));
      }
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
      watermarkText: context.config.previewWatermarkText
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
