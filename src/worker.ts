import crypto from "node:crypto";
import {
  BlobStoreLike,
  HandlerContext,
  JobType,
  QUEUE_PREFIXES,
  QueueBlob,
  QueueRecord,
  StatusRecord,
  WorkerConfig,
  WorkerStores
} from "./types";
import { executeJob } from "./handlers";
import { PipelineJobError } from "./pythonBridge";
import { calculateRetryDelay } from "./retry";

function nowIso(): string {
  return new Date().toISOString();
}

function encodeKey(value: string): string {
  return Buffer.from(value).toString("base64url");
}

function statusKey(queueKeyValue: string): string {
  return `status/${encodeKey(queueKeyValue)}.json`;
}

function lockKey(queueKeyValue: string): string {
  return `lock/${encodeKey(queueKeyValue)}.json`;
}

function deadLetterKey(queueKeyValue: string): string {
  return `dead/${encodeKey(queueKeyValue)}.json`;
}

function queueIdentifier(record: QueueRecord<JobType>): string {
  const payload = record.payload as Record<string, unknown>;
  const knownId = payload.uploadId || payload.sessionId;
  return String(knownId || crypto.randomUUID().slice(0, 8));
}

function queueRetryKey(record: QueueRecord<JobType>, attempt: number): string {
  const timestamp = nowIso().replace(/[:.]/g, "-");
  return `${record.type}/${timestamp}-${queueIdentifier(record)}-attempt-${attempt}.json`;
}

async function statusRecordFor(
  store: BlobStoreLike,
  queueKeyValue: string
): Promise<StatusRecord | null> {
  const existing = await store.getWithMetadata<StatusRecord>(statusKey(queueKeyValue), { type: "json" });
  return existing?.data ?? null;
}

async function writeStatus(
  stores: WorkerStores,
  queueKeyValue: string,
  value: StatusRecord
): Promise<void> {
  await stores.status.setJSON(statusKey(queueKeyValue), value);
}

async function claimLock(
  stores: WorkerStores,
  queueKeyValue: string,
  workerId: string,
  lockTtlMs: number
): Promise<boolean> {
  const key = lockKey(queueKeyValue);
  const expiresAt = new Date(Date.now() + lockTtlMs).toISOString();
  const value = { workerId, queueKey: queueKeyValue, claimedAt: nowIso(), expiresAt };

  const created = await stores.locks.setJSON(key, value, { onlyIfNew: true });
  if (created.modified) {
    return true;
  }

  const existing = await stores.locks.getWithMetadata<{ expiresAt?: string }>(key, { type: "json" });
  if (!existing?.data?.expiresAt) {
    return false;
  }

  if (Date.parse(existing.data.expiresAt) > Date.now()) {
    return false;
  }

  const refreshed = await stores.locks.setJSON(key, value, {
    onlyIfMatch: existing.etag
  });

  return refreshed.modified;
}

async function releaseLock(stores: WorkerStores, queueKeyValue: string): Promise<void> {
  await stores.locks.delete(lockKey(queueKeyValue));
}

export async function listCandidateJobs(
  queueStore: BlobStoreLike,
  maxJobs: number
): Promise<QueueBlob[]> {
  const blobs: QueueBlob[] = [];

  for (const prefix of QUEUE_PREFIXES) {
    for await (const page of queueStore.list({ prefix, paginate: true })) {
      blobs.push(...page.blobs);
    }
  }

  const sortableToken = (key: string): string => key.split("/")[1] || key;
  blobs.sort((left, right) => sortableToken(left.key).localeCompare(sortableToken(right.key)));
  return blobs.slice(0, Math.max(maxJobs, blobs.length));
}

export async function processQueueBlob(
  blob: QueueBlob,
  context: HandlerContext
): Promise<"processed" | "skipped"> {
  const existingStatus = await statusRecordFor(context.stores.status, blob.key);
  if (
    existingStatus &&
    ["completed", "retry_scheduled", "dead_lettered", "skipped"].includes(existingStatus.status)
  ) {
    return "skipped";
  }

  const lockClaimed = await claimLock(
    context.stores,
    blob.key,
    context.config.workerId,
    context.config.lockTtlMs
  );
  if (!lockClaimed) {
    return "skipped";
  }

  try {
    const queueEntry = await context.stores.queue.getWithMetadata<QueueRecord<JobType>>(blob.key, {
      type: "json"
    });
    if (!queueEntry) {
      await writeStatus(context.stores, blob.key, {
        queueKey: blob.key,
        type: "upload_photo",
        workerId: context.config.workerId,
        status: "skipped",
        attempts: 0,
        updatedAt: nowIso(),
        error: "Queue record missing"
      });
      return "skipped";
    }

    const record = queueEntry.data;
    const attempts = record.attempt ?? 1;

    if (record.notBefore && Date.parse(record.notBefore) > Date.now()) {
      await writeStatus(context.stores, blob.key, {
        queueKey: blob.key,
        type: record.type,
        workerId: context.config.workerId,
        status: "skipped",
        attempts,
        updatedAt: nowIso(),
        nextAttemptAt: record.notBefore
      });
      return "skipped";
    }

    await writeStatus(context.stores, blob.key, {
      queueKey: blob.key,
      type: record.type,
      workerId: context.config.workerId,
      status: "processing",
      attempts,
      updatedAt: nowIso()
    });

    try {
      const { result, resultKey } = await executeJob(record.type, record.payload as never, context);
      await context.stores.results.setJSON(resultKey, result);
      await writeStatus(context.stores, blob.key, {
        queueKey: blob.key,
        type: record.type,
        workerId: context.config.workerId,
        status: "completed",
        attempts,
        updatedAt: nowIso(),
        resultKey
      });

      return "processed";
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const errorCode = error instanceof PipelineJobError ? error.code : undefined;
      const retryable = !(error instanceof PipelineJobError) || error.retryable;
      if (error instanceof PipelineJobError && error.details) {
        process.stdout.write(
          `[rizzup-worker] preview-face-check-error-details ${JSON.stringify({
            queueKey: blob.key,
            errorCode,
            details: error.details
          })}\n`
        );
      }

      if (retryable && attempts < context.config.maxAttempts) {
        const nextAttempt = attempts + 1;
        const delayMs = calculateRetryDelay(
          attempts,
          context.config.retryBaseDelayMs,
          context.config.retryMaxDelayMs
        );
        const nextAttemptAt = new Date(Date.now() + delayMs).toISOString();

        const retryRecord: QueueRecord<JobType> = {
          ...record,
          queuedAt: nowIso(),
          attempt: nextAttempt,
          notBefore: nextAttemptAt,
          context: {
            ...record.context,
            retryOf: blob.key,
            lastError: message
          }
        };

        await context.stores.queue.setJSON(queueRetryKey(record, nextAttempt), retryRecord, {
          onlyIfNew: true
        });
        await writeStatus(context.stores, blob.key, {
          queueKey: blob.key,
          type: record.type,
          workerId: context.config.workerId,
          status: "retry_scheduled",
          attempts,
          updatedAt: nowIso(),
          nextAttemptAt,
          error: message,
          errorCode
        });
        return "processed";
      }

      await context.stores.deadLetter.setJSON(deadLetterKey(blob.key), {
        queueKey: blob.key,
        failedAt: nowIso(),
        errorCode,
        error: message,
        record
      });
      await writeStatus(context.stores, blob.key, {
        queueKey: blob.key,
        type: record.type,
        workerId: context.config.workerId,
        status: "dead_lettered",
        attempts,
        updatedAt: nowIso(),
        error: message,
        errorCode
      });
      return "processed";
    }
  } finally {
    await releaseLock(context.stores, blob.key);
  }
}

export async function pollOnce(config: WorkerConfig, stores: WorkerStores): Promise<number> {
  const jobs = await listCandidateJobs(stores.queue, config.maxJobsPerPoll);
  const context: HandlerContext = { config, stores };
  let processed = 0;

  for (const blob of jobs) {
    if (processed >= config.maxJobsPerPoll) {
      break;
    }

    const result = await processQueueBlob(blob, context);
    if (result === "processed") {
      processed += 1;
    }
  }

  return processed;
}

export async function runWorker(config: WorkerConfig, stores: WorkerStores): Promise<void> {
  while (true) {
    try {
      await pollOnce(config, stores);
    } catch (error) {
      const message = error instanceof Error ? error.stack || error.message : String(error);
      process.stderr.write(`[rizzup-worker] ${message}\n`);
    }

    await new Promise((resolve) => setTimeout(resolve, config.pollIntervalMs));
  }
}
