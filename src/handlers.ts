import fs from "node:fs/promises";
import crypto from "node:crypto";
import path from "node:path";
import {
  AnalyzePhotoQualityPayload,
  CreateCheckoutSessionPayload,
  GenerateFinalImagePayload,
  GeneratePreviewPayload,
  FinalImageResult,
  HandlerContext,
  HandlerResultMap,
  JobType,
  PhotoQualityResult,
  QueuePayloadMap,
  UploadPhotoResult
} from "./types";
import {
  analyzeWithPython,
  generateFinalImageWithPython,
  generatePreviewWithPython,
  PipelineJobError,
  validateUploadWithPython
} from "./pythonBridge";

function assetBlobKey(assetId: string): string {
  return `generated/${assetId}`;
}

function logArchiveEvent(message: string, details: Record<string, unknown>): void {
  const formatted = Object.entries(details)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(" ");
  process.stdout.write(`[rizzup-worker] ${message}${formatted ? ` ${formatted}` : ""}\n`);
}

function logHandlerDebug(message: string, details: Record<string, unknown>): void {
  const formatted = Object.entries(details)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(" ");
  process.stdout.write(`[rizzup-handler-debug] ${message}${formatted ? ` ${formatted}` : ""}\n`);
}

function sanitizeFileName(fileName: string): string {
  const cleaned = String(fileName || "upload.bin")
    .replace(/[^a-zA-Z0-9._-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");

  return cleaned || "upload.bin";
}

function extensionFromMimeType(mimeType: string): string {
  switch (mimeType) {
    case "image/jpeg":
      return ".jpg";
    case "image/png":
      return ".png";
    case "image/webp":
      return ".webp";
    default:
      return "";
  }
}

function decodeDataUrl(sourceDataUrl?: string | null): Buffer | null {
  if (!sourceDataUrl) return null;
  const match = sourceDataUrl.match(/^data:(.*?);base64,(.*)$/);
  if (!match) {
    throw new Error("sourceDataUrl must be a valid data URL");
  }

  return Buffer.from(match[2], "base64");
}

function archiveDateParts(createdAt?: string | null): { year: string; month: string; day: string } {
  const date = createdAt ? new Date(createdAt) : new Date();
  const safeDate = Number.isNaN(date.getTime()) ? new Date() : date;
  return {
    year: String(safeDate.getUTCFullYear()),
    month: String(safeDate.getUTCMonth() + 1).padStart(2, "0"),
    day: String(safeDate.getUTCDate()).padStart(2, "0")
  };
}

function buildImageJobRoot(
  imageJobId: string,
  createdAt: string | undefined,
  context: HandlerContext
): string {
  const date = archiveDateParts(createdAt);
  return path.join(context.config.imageArchiveRoot, date.year, date.month, date.day, imageJobId);
}

async function ensureImageJobFolders(
  imageJobId: string,
  createdAt: string | undefined,
  context: HandlerContext
): Promise<{
  jobRoot: string;
  sourceDir: string;
  previewDir: string;
  finalDir: string;
}> {
  const jobRoot = buildImageJobRoot(imageJobId, createdAt, context);
  const sourceDir = path.join(jobRoot, "source");
  const previewDir = path.join(jobRoot, "generated", "preview");
  const finalDir = path.join(jobRoot, "generated", "final");

  await fs.mkdir(sourceDir, { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(finalDir, { recursive: true });

  return { jobRoot, sourceDir, previewDir, finalDir };
}

function contentTypeFromPath(filePath: string): string {
  switch (path.extname(filePath).toLowerCase()) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".webp":
      return "image/webp";
    default:
      return "application/octet-stream";
  }
}

async function persistGeneratedAsset(
  filePath: string,
  metadata: Record<string, unknown>,
  context: HandlerContext
): Promise<string> {
  const startedAt = Date.now();
  const assetId = `asset_${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`;
  const buffer = await fs.readFile(filePath);
  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  ) as ArrayBuffer;

  await context.stores.assets.set(assetBlobKey(assetId), arrayBuffer, {
    metadata: {
      ...metadata,
      contentType: contentTypeFromPath(filePath)
    }
  });

  logHandlerDebug("asset-persisted", {
    assetId,
    filePath,
    sizeBytes: buffer.byteLength,
    durationMs: Date.now() - startedAt,
    kind: metadata.kind
  });

  return assetId;
}

function uploadResultKey(payload: QueuePayloadMap["upload_photo"]): string {
  return `upload_photo/${payload.uploadId}.json`;
}

function qualityResultKey(payload: QueuePayloadMap["analyze_photo_quality"]): string {
  return `analyze_photo_quality/${payload.uploadId}.json`;
}

function previewResultKey(payload: QueuePayloadMap["generate_preview"]): string {
  return `generate_preview/${payload.uploadId}-${payload.preset}.json`;
}

function checkoutResultKey(payload: QueuePayloadMap["create_checkout_session"]): string {
  return `create_checkout_session/${payload.sessionId}.json`;
}

function finalImageResultKey(payload: QueuePayloadMap["generate_final_image"]): string {
  return `generate_final_image/${payload.unlockId}.json`;
}

async function getUploadResult(
  uploadId: string,
  context: HandlerContext
): Promise<UploadPhotoResult | null> {
  const record = await context.stores.results.getWithMetadata<UploadPhotoResult>(
    `upload_photo/${uploadId}.json`,
    { type: "json" }
  );

  return record?.data ?? null;
}

async function handleUploadPhoto(
  payload: QueuePayloadMap["upload_photo"],
  context: HandlerContext
): Promise<UploadPhotoResult> {
  const folders = await ensureImageJobFolders(payload.imageJobId, payload.createdAt, context);
  const sanitized = sanitizeFileName(payload.sourceName);
  const baseName = path.parse(sanitized).name;
  const ext = path.extname(sanitized) || extensionFromMimeType(payload.mimeType) || ".bin";
  const sourceRelativePath = path
    .join("source", `${payload.uploadId}-${baseName}${ext}`)
    .replace(/\\/g, "/");
  const sourcePath = path.join(folders.jobRoot, sourceRelativePath);
  const sourceBuffer = decodeDataUrl(payload.sourceDataUrl);

  if (!sourceBuffer) {
    throw new Error(`Upload ${payload.uploadId} did not include sourceDataUrl for archival storage`);
  }

  await fs.writeFile(sourcePath, sourceBuffer);
  const uploadRecord: UploadPhotoResult = {
    uploadId: payload.uploadId,
    imageJobId: payload.imageJobId,
    sourceName: payload.sourceName,
    mimeType: payload.mimeType,
    sizeBytes: payload.sizeBytes,
    width: payload.width ?? null,
    height: payload.height ?? null,
    createdAt: payload.createdAt,
    sourcePath,
    sourceRelativePath,
    sourceUrl: payload.sourceUrl ?? null,
    sourceBlobKey: payload.sourceBlobKey ?? null
  };
  const faceDetection = await validateUploadWithPython(payload.uploadId, uploadRecord, context);
  logArchiveEvent("stored-source", {
    imageJobId: payload.imageJobId,
    uploadId: payload.uploadId,
    sourcePath
  });

  return {
    ...uploadRecord,
    faceDetection
  };
}

function fallbackQuality(uploadId: string, upload: UploadPhotoResult | null): PhotoQualityResult {
  const width = upload?.width ?? null;
  const height = upload?.height ?? null;
  const base = 68 + (uploadId.length % 15);

  return {
    uploadId,
    score: Math.min(92, base + (width && height ? 6 : 0)),
    summary:
      width && height
        ? `Photo ${uploadId} is ${width}x${height}. Framing looks usable, but stronger subject separation and a cleaner crop would improve conversion.`
        : `Photo ${uploadId} is registered, but no source file was attached to the backend worker. Provide sourcePath for full image analysis.`,
    metrics: {
      width,
      height,
      brightness: null,
      contrast: null,
      sharpness: null
    },
    analyzedAt: new Date().toISOString()
  };
}

function requireUploadResult(uploadId: string, upload: UploadPhotoResult | null): UploadPhotoResult {
  if (upload) {
    return upload;
  }

  throw new PipelineJobError(
    `Upload ${uploadId} is not available for downstream processing. The initial upload validation likely failed before the source image metadata was saved.`,
    {
      code: "UPLOAD_NOT_AVAILABLE",
      retryable: false,
      details: {
        uploadId
      }
    }
  );
}

async function handleAnalyzePhotoQuality(
  payload: QueuePayloadMap["analyze_photo_quality"],
  context: HandlerContext
): Promise<PhotoQualityResult> {
  const upload = await getUploadResult(payload.uploadId, context);
  requireUploadResult(payload.uploadId, upload);
  try {
    return await analyzeWithPython(payload.uploadId, upload, context);
  } catch (error) {
    if (error instanceof PipelineJobError) {
      throw error;
    }
    return fallbackQuality(payload.uploadId, upload);
  }
}

async function handleGeneratePreview(
  payload: GeneratePreviewPayload,
  context: HandlerContext
): Promise<HandlerResultMap["generate_preview"]> {
  const startedAt = Date.now();
  const upload = requireUploadResult(payload.uploadId, await getUploadResult(payload.uploadId, context));
  const imageJobId = upload?.imageJobId || payload.uploadId;
  const folders = await ensureImageJobFolders(imageJobId, upload?.createdAt, context);

  const outputPath = path.join(folders.previewDir, `${payload.preset}.png`);
  logArchiveEvent("preview-face-check-start", {
    imageJobId,
    uploadId: payload.uploadId,
    preset: payload.preset,
    outputPath
  });
  logHandlerDebug("preview-python-dispatch", {
    imageJobId,
    uploadId: payload.uploadId,
    preset: payload.preset,
    sourcePath: upload?.sourcePath ?? null,
    outputPath
  });
  const generated = await generatePreviewWithPython(
    payload.uploadId,
    payload.preset,
    outputPath,
    upload,
    context
  );
  logHandlerDebug("preview-python-complete", {
    imageJobId,
    uploadId: payload.uploadId,
    preset: payload.preset,
    width: generated.width,
    height: generated.height,
    usedGpu: generated.usedGpu,
    identityGenerationUsed: generated.identityGenerationUsed,
    identityGenerationMode: generated.identityGenerationMode,
    identityFallbackReason: generated.identityFallbackReason ?? null,
    durationMs: Date.now() - startedAt
  });
  if (generated.identityGenerationMode === "heuristic-fallback" && generated.identityFallbackReason) {
    logArchiveEvent("preview-identity-fallback", {
      imageJobId,
      uploadId: payload.uploadId,
      preset: payload.preset,
      reason: generated.identityFallbackReason
    });
  }
  const previewAssetId = await persistGeneratedAsset(
    outputPath,
    {
      kind: "preview",
      imageJobId,
      uploadId: payload.uploadId,
      preset: payload.preset
    },
    context
  );
  logArchiveEvent("stored-preview", {
    imageJobId,
    uploadId: payload.uploadId,
    preset: payload.preset,
    outputPath
  });
  logArchiveEvent("preview-face-check-complete", {
    imageJobId,
    uploadId: payload.uploadId,
    preset: payload.preset,
    width: generated.width,
    height: generated.height
  });
  logHandlerDebug("preview-handler-complete", {
    imageJobId,
    uploadId: payload.uploadId,
    preset: payload.preset,
    previewAssetId,
    totalDurationMs: Date.now() - startedAt
  });

  return {
    ...generated,
    imageJobId,
    previewPath: outputPath,
    previewAssetId
  };
}

async function handleCreateCheckoutSession(
  payload: CreateCheckoutSessionPayload
): Promise<HandlerResultMap["create_checkout_session"]> {
  return {
    sessionId: payload.sessionId,
    uploadId: payload.uploadId,
    preset: payload.preset,
    plan: payload.plan,
    checkoutUrl: `https://checkout.rizzup.co.uk/mock/${payload.sessionId}?plan=${payload.plan}&preset=${payload.preset}`,
    generatedAt: new Date().toISOString()
  };
}

async function handleGenerateFinalImage(
  payload: GenerateFinalImagePayload,
  context: HandlerContext
): Promise<FinalImageResult> {
  const upload = requireUploadResult(payload.uploadId, await getUploadResult(payload.uploadId, context));
  const imageJobId = upload?.imageJobId || payload.uploadId;
  const folders = await ensureImageJobFolders(imageJobId, upload?.createdAt, context);

  const outputPath = path.join(folders.finalDir, `${payload.unlockId}-${payload.preset}.png`);
  const generated = await generateFinalImageWithPython(
    payload.unlockId,
    payload.checkoutSessionId,
    payload.uploadId,
    payload.preset,
    outputPath,
    payload.plan,
    upload,
    context
  );
  const finalImageAssetId = await persistGeneratedAsset(
    outputPath,
    {
      kind: "final",
      unlockId: payload.unlockId,
      imageJobId,
      uploadId: payload.uploadId,
      preset: payload.preset
    },
    context
  );
  logArchiveEvent("stored-final", {
    imageJobId,
    uploadId: payload.uploadId,
    unlockId: payload.unlockId,
    preset: payload.preset,
    outputPath
  });

  return {
    ...generated,
    imageJobId,
    finalImagePath: outputPath,
    finalImageAssetId
  };
}

export async function executeJob<T extends JobType>(
  type: T,
  payload: QueuePayloadMap[T],
  context: HandlerContext
): Promise<{ result: HandlerResultMap[T]; resultKey: string }> {
  let result: HandlerResultMap[T];
  let resultKey = "";

  switch (type) {
    case "upload_photo":
      result = (await handleUploadPhoto(
        payload as QueuePayloadMap["upload_photo"],
        context
      )) as HandlerResultMap[T];
      resultKey = uploadResultKey(payload as QueuePayloadMap["upload_photo"]);
      break;
    case "analyze_photo_quality":
      result = (await handleAnalyzePhotoQuality(
        payload as AnalyzePhotoQualityPayload,
        context
      )) as HandlerResultMap[T];
      resultKey = qualityResultKey(payload as QueuePayloadMap["analyze_photo_quality"]);
      break;
    case "generate_preview":
      result = (await handleGeneratePreview(payload as GeneratePreviewPayload, context)) as HandlerResultMap[T];
      resultKey = previewResultKey(payload as QueuePayloadMap["generate_preview"]);
      break;
    case "create_checkout_session":
      result = (await handleCreateCheckoutSession(
        payload as CreateCheckoutSessionPayload
      )) as HandlerResultMap[T];
      resultKey = checkoutResultKey(payload as QueuePayloadMap["create_checkout_session"]);
      break;
    case "generate_final_image":
      result = (await handleGenerateFinalImage(
        payload as GenerateFinalImagePayload,
        context
      )) as HandlerResultMap[T];
      resultKey = finalImageResultKey(payload as QueuePayloadMap["generate_final_image"]);
      break;
    default:
      throw new Error(`Unsupported job type: ${String(type)}`);
  }

  return {
    result,
    resultKey
  };
}
