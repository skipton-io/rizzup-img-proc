export const QUEUE_PREFIXES = [
  "upload_photo/",
  "analyze_photo_quality/",
  "generate_preview/",
  "create_checkout_session/",
  "generate_final_image/"
] as const;

export type JobType =
  | "upload_photo"
  | "analyze_photo_quality"
  | "generate_preview"
  | "create_checkout_session"
  | "generate_final_image";

export type UploadPhotoPayload = {
  uploadId: string;
  sourceName: string;
  mimeType: string;
  sizeBytes: number;
  width?: number | null;
  height?: number | null;
  lastModified?: number | null;
  createdAt: string;
  sourcePath?: string | null;
  sourceUrl?: string | null;
  sourceBlobKey?: string | null;
};

export type AnalyzePhotoQualityPayload = {
  uploadId: string;
  requestedAt: string;
};

export type GeneratePreviewPayload = {
  uploadId: string;
  preset: "natural" | "professional" | "lifestyle" | "fitness" | "travel";
  requestedAt: string;
};

export type CreateCheckoutSessionPayload = {
  sessionId: string;
  uploadId: string;
  preset: string;
  plan: "1_photo" | "5_photos" | "10_photos";
  requestedAt: string;
};

export type GenerateFinalImagePayload = {
  unlockId: string;
  uploadId: string;
  preset: "natural" | "professional" | "lifestyle" | "fitness" | "travel";
  plan: "1_photo" | "5_photos" | "10_photos";
  checkoutSessionId: string;
  requestedAt: string;
};

export type QueuePayloadMap = {
  upload_photo: UploadPhotoPayload;
  analyze_photo_quality: AnalyzePhotoQualityPayload;
  generate_preview: GeneratePreviewPayload;
  create_checkout_session: CreateCheckoutSessionPayload;
  generate_final_image: GenerateFinalImagePayload;
};

export type QueueRecord<T extends JobType = JobType> = {
  type: T;
  queuedAt: string;
  payload: QueuePayloadMap[T];
  context?: {
    userAgent?: string;
    netlifyId?: string;
    requestId?: string;
    retryOf?: string;
    lastError?: string;
  };
  attempt?: number;
  notBefore?: string;
};

export type QueueBlob = {
  key: string;
  etag?: string;
};

export type JsonBlobResult<T> = {
  data: T;
  etag?: string;
  metadata?: Record<string, unknown>;
};

export type BlobPage = {
  blobs: QueueBlob[];
};

export interface BlobStoreLike {
  list(options: { prefix: string; paginate: true }): AsyncIterable<BlobPage>;
  getWithMetadata<T>(key: string, options: { type: "json" }): Promise<JsonBlobResult<T> | null>;
  setJSON<T>(
    key: string,
    value: T,
    options?: {
      metadata?: Record<string, unknown>;
      onlyIfMatch?: string;
      onlyIfNew?: boolean;
    }
  ): Promise<{ modified: boolean; etag?: string }>;
  delete(key: string): Promise<void>;
}

export type WorkerStatus =
  | "processing"
  | "completed"
  | "retry_scheduled"
  | "dead_lettered"
  | "skipped";

export type StatusRecord = {
  queueKey: string;
  type: JobType;
  workerId: string;
  status: WorkerStatus;
  attempts: number;
  updatedAt: string;
  resultKey?: string;
  nextAttemptAt?: string;
  error?: string;
};

export type UploadPhotoResult = {
  uploadId: string;
  sourceName: string;
  mimeType: string;
  sizeBytes: number;
  width?: number | null;
  height?: number | null;
  createdAt: string;
  sourcePath?: string | null;
  sourceUrl?: string | null;
  sourceBlobKey?: string | null;
};

export type PhotoQualityResult = {
  uploadId: string;
  score: number;
  summary: string;
  metrics: {
    width?: number | null;
    height?: number | null;
    brightness?: number | null;
    contrast?: number | null;
    sharpness?: number | null;
  };
  analyzedAt: string;
};

export type PreviewResult = {
  uploadId: string;
  preset: string;
  previewPath: string;
  previewUrl: string;
  watermarkText: string;
  usedGpu: boolean;
  width: number;
  height: number;
  generatedAt: string;
};

export type CheckoutResult = {
  sessionId: string;
  uploadId: string;
  preset: string;
  plan: string;
  checkoutUrl: string;
  generatedAt: string;
};

export type FinalImageResult = {
  unlockId: string;
  checkoutSessionId: string;
  uploadId: string;
  preset: string;
  plan: string;
  finalImagePath: string;
  finalImageUrl: string;
  usedGpu: boolean;
  width: number;
  height: number;
  generatedAt: string;
};

export type HandlerResultMap = {
  upload_photo: UploadPhotoResult;
  analyze_photo_quality: PhotoQualityResult;
  generate_preview: PreviewResult;
  create_checkout_session: CheckoutResult;
  generate_final_image: FinalImageResult;
};

export type WorkerStores = {
  queue: BlobStoreLike;
  status: BlobStoreLike;
  results: BlobStoreLike;
  locks: BlobStoreLike;
  deadLetter: BlobStoreLike;
};

export type WorkerConfig = {
  netlifySiteId: string;
  netlifyAccessToken: string;
  queueStore: string;
  statusStore: string;
  resultsStore: string;
  locksStore: string;
  deadLetterStore: string;
  pollIntervalMs: number;
  maxJobsPerPoll: number;
  lockTtlMs: number;
  maxAttempts: number;
  retryBaseDelayMs: number;
  retryMaxDelayMs: number;
  workerId: string;
  previewWatermarkText: string;
  resultsDir: string;
  sourceImageRoot?: string;
  resultsPublicBaseUrl?: string;
  pythonExecutable: string;
  pythonScript: string;
};

export type HandlerContext = {
  config: WorkerConfig;
  stores: WorkerStores;
};
