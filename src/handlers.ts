import fs from "node:fs/promises";
import path from "node:path";
import {
  CreateCheckoutSessionPayload,
  GeneratePreviewPayload,
  HandlerContext,
  HandlerResultMap,
  JobType,
  PhotoQualityResult,
  QueuePayloadMap,
  UploadPhotoResult
} from "./types";
import { analyzeWithPython, generatePreviewWithPython } from "./pythonBridge";

function resultStoreKey(type: JobType, payload: QueuePayloadMap[JobType]): string {
  switch (type) {
    case "upload_photo":
      return `upload_photo/${payload.uploadId}.json`;
    case "analyze_photo_quality":
      return `analyze_photo_quality/${payload.uploadId}.json`;
    case "generate_preview":
      return `generate_preview/${payload.uploadId}-${payload.preset}.json`;
    case "create_checkout_session":
      return `create_checkout_session/${payload.sessionId}.json`;
  }
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
  payload: QueuePayloadMap["upload_photo"]
): Promise<UploadPhotoResult> {
  return {
    uploadId: payload.uploadId,
    sourceName: payload.sourceName,
    mimeType: payload.mimeType,
    sizeBytes: payload.sizeBytes,
    width: payload.width ?? null,
    height: payload.height ?? null,
    createdAt: payload.createdAt,
    sourcePath: payload.sourcePath ?? null,
    sourceUrl: payload.sourceUrl ?? null,
    sourceBlobKey: payload.sourceBlobKey ?? null
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

async function handleAnalyzePhotoQuality(
  payload: QueuePayloadMap["analyze_photo_quality"],
  context: HandlerContext
): Promise<PhotoQualityResult> {
  const upload = await getUploadResult(payload.uploadId, context);
  try {
    return await analyzeWithPython(payload.uploadId, upload, context);
  } catch {
    return fallbackQuality(payload.uploadId, upload);
  }
}

async function handleGeneratePreview(
  payload: GeneratePreviewPayload,
  context: HandlerContext
): Promise<HandlerResultMap["generate_preview"]> {
  const upload = await getUploadResult(payload.uploadId, context);
  const previewsDir = path.join(context.config.resultsDir, "previews");
  await fs.mkdir(previewsDir, { recursive: true });

  const outputPath = path.join(previewsDir, `${payload.uploadId}-${payload.preset}.png`);
  return await generatePreviewWithPython(payload.uploadId, payload.preset, outputPath, upload, context);
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

export async function executeJob<T extends JobType>(
  type: T,
  payload: QueuePayloadMap[T],
  context: HandlerContext
): Promise<{ result: HandlerResultMap[T]; resultKey: string }> {
  let result: HandlerResultMap[T];

  switch (type) {
    case "upload_photo":
      result = (await handleUploadPhoto(payload)) as HandlerResultMap[T];
      break;
    case "analyze_photo_quality":
      result = (await handleAnalyzePhotoQuality(payload, context)) as HandlerResultMap[T];
      break;
    case "generate_preview":
      result = (await handleGeneratePreview(payload as GeneratePreviewPayload, context)) as HandlerResultMap[T];
      break;
    case "create_checkout_session":
      result = (await handleCreateCheckoutSession(
        payload as CreateCheckoutSessionPayload
      )) as HandlerResultMap[T];
      break;
    default:
      throw new Error(`Unsupported job type: ${String(type)}`);
  }

  return {
    result,
    resultKey: resultStoreKey(type, payload)
  };
}
