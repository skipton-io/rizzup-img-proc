import crypto from "node:crypto";
import {
  ArchiveStorage,
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

function logWorkerEvent(event: string, details: Record<string, unknown>): void {
  const formatted = Object.entries(details)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(" ");
  process.stdout.write(`[rizzup-worker-debug] ${event}${formatted ? ` ${formatted}` : ""}\n`);
}

function queueIdentifier(record: QueueRecord<JobType>): string {
  const payload = record.payload as Record<string, unknown>;
  const knownId =
    payload.unlockId ||
    payload.renderKey ||
    (payload.uploadId && payload.preset
      ? `${String(payload.uploadId)}-${String(payload.preset)}`
      : undefined) ||
    payload.sessionId ||
    payload.uploadId;
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

async function writeStatusAliases(
  stores: WorkerStores,
  queueKeys: string[],
  value: StatusRecord
): Promise<void> {
  const uniqueKeys = [...new Set(queueKeys.filter(Boolean))];
  await Promise.all(
    uniqueKeys.map((queueKeyValue) =>
      writeStatus(stores, queueKeyValue, {
        ...value,
        queueKey: queueKeyValue
      })
    )
  );
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

async function deleteQueueBlob(stores: WorkerStores, queueKeyValue: string): Promise<void> {
  await stores.queue.delete(queueKeyValue);
}

export async function listCandidateJobs(
  queueStore: BlobStoreLike,
  maxJobs: number
): Promise<QueueBlob[]> {
  const blobs: Array<QueueBlob & { queuedAt: string; index: number }> = [];
  let index = 0;

  for (const prefix of QUEUE_PREFIXES) {
    for await (const page of queueStore.list({ prefix, paginate: true })) {
      for (const blob of page.blobs) {
        const queueEntry = await queueStore.getWithMetadata<QueueRecord<JobType>>(blob.key, {
          type: "json"
        });
        blobs.push({
          ...blob,
          queuedAt: queueEntry?.data?.queuedAt || "",
          index
        });
        index += 1;
      }
    }
  }

  blobs.sort((left, right) => {
    const queuedAtCompare = left.queuedAt.localeCompare(right.queuedAt);
    if (queuedAtCompare !== 0) {
      return queuedAtCompare;
    }

    return left.index - right.index;
  });

  return blobs.map(({ queuedAt: _queuedAt, index: _index, ...blob }) => blob);
}

export async function processQueueBlob(
  blob: QueueBlob,
  context: HandlerContext
): Promise<"processed" | "skipped"> {
  const startedAt = Date.now();
  const existingStatus = await statusRecordFor(context.stores.status, blob.key);
  if (
    existingStatus &&
    ["completed", "retry_scheduled", "dead_lettered", "skipped"].includes(existingStatus.status)
  ) {
    logWorkerEvent("skip-existing-status", {
      queueKey: blob.key,
      status: existingStatus.status,
      attempts: existingStatus.attempts
    });
    return "skipped";
  }

  const lockClaimed = await claimLock(
    context.stores,
    blob.key,
    context.config.workerId,
    context.config.lockTtlMs
  );
  if (!lockClaimed) {
    logWorkerEvent("skip-lock-not-claimed", {
      queueKey: blob.key,
      workerId: context.config.workerId
    });
    return "skipped";
  }

  try {
    const queueEntry = await context.stores.queue.getWithMetadata<QueueRecord<JobType>>(blob.key, {
      type: "json"
    });
    if (!queueEntry) {
      logWorkerEvent("queue-record-missing", {
        queueKey: blob.key
      });
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
    const rootQueueKey = record.context?.rootQueueKey || record.context?.retryOf || blob.key;
    const statusQueueKeys = [blob.key, rootQueueKey];
    logWorkerEvent("job-start", {
      queueKey: blob.key,
      type: record.type,
      attempts,
      workerId: context.config.workerId,
      notBefore: record.notBefore,
      retryOf: record.context?.retryOf
    });

    if (record.notBefore && Date.parse(record.notBefore) > Date.now()) {
      logWorkerEvent("job-skipped-not-before", {
        queueKey: blob.key,
        type: record.type,
        attempts,
        notBefore: record.notBefore
      });
      await writeStatusAliases(context.stores, statusQueueKeys, {
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

    await writeStatusAliases(context.stores, statusQueueKeys, {
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
      logWorkerEvent("job-complete", {
        queueKey: blob.key,
        type: record.type,
        attempts,
        durationMs: Date.now() - startedAt,
        resultKey
      });
      await writeStatusAliases(context.stores, statusQueueKeys, {
        queueKey: blob.key,
        type: record.type,
        workerId: context.config.workerId,
        status: "completed",
        attempts,
        updatedAt: nowIso(),
        resultKey
      });
      await deleteQueueBlob(context.stores, blob.key);

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
            rootQueueKey,
            lastError: message
          }
        };

        const retryKey = queueRetryKey(record, nextAttempt);
        await context.stores.queue.setJSON(retryKey, retryRecord, {
          onlyIfNew: true
        });
        logWorkerEvent("job-retry-scheduled", {
          queueKey: blob.key,
          type: record.type,
          attempts,
          nextAttempt,
          retryKey,
          nextAttemptAt,
          delayMs,
          errorCode,
          error: message
        });
        await writeStatusAliases(context.stores, statusQueueKeys, {
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
        await deleteQueueBlob(context.stores, blob.key);
        return "processed";
      }

      await context.stores.deadLetter.setJSON(deadLetterKey(blob.key), {
        queueKey: blob.key,
        failedAt: nowIso(),
        errorCode,
        error: message,
        record
      });
      logWorkerEvent("job-dead-lettered", {
        queueKey: blob.key,
        type: record.type,
        attempts,
        durationMs: Date.now() - startedAt,
        errorCode,
        error: message
      });
      await writeStatusAliases(context.stores, statusQueueKeys, {
        queueKey: blob.key,
        type: record.type,
        workerId: context.config.workerId,
        status: "dead_lettered",
        attempts,
        updatedAt: nowIso(),
        error: message,
        errorCode
      });
      await deleteQueueBlob(context.stores, blob.key);
      return "processed";
    }
  } finally {
    logWorkerEvent("job-finish", {
      queueKey: blob.key,
      durationMs: Date.now() - startedAt
    });
    await releaseLock(context.stores, blob.key);
  }
}

export async function pollWithArchiveStorage(
  config: WorkerConfig,
  stores: WorkerStores,
  archiveStorage: ArchiveStorage
): Promise<number> {
  const pollStartedAt = Date.now();
  const jobs = await listCandidateJobs(stores.queue, config.maxJobsPerPoll);
  const context: HandlerContext = { config, stores, archiveStorage };
  let processed = 0;
  logWorkerEvent("poll-start", {
    workerId: config.workerId,
    candidateJobs: jobs.length,
    maxJobsPerPoll: config.maxJobsPerPoll
  });

  for (const blob of jobs) {
    if (processed >= config.maxJobsPerPoll) {
      break;
    }

    const result = await processQueueBlob(blob, context);
    if (result === "processed") {
      processed += 1;
    }
  }

  logWorkerEvent("poll-complete", {
    workerId: config.workerId,
    processed,
    candidateJobs: jobs.length,
    durationMs: Date.now() - pollStartedAt
  });

  return processed;
}

export async function pollOnce(
  config: WorkerConfig,
  stores: WorkerStores,
  archiveStorage: ArchiveStorage
): Promise<number> {
  return await pollWithArchiveStorage(config, stores, archiveStorage);
}

export async function runWorker(
  config: WorkerConfig,
  stores: WorkerStores,
  archiveStorage: ArchiveStorage
): Promise<void> {
  const startedAt = Date.now();

  while (Date.now() - startedAt < config.maxRuntimeMs) {
    try {
      await pollWithArchiveStorage(config, stores, archiveStorage);
    } catch (error) {
      const message = error instanceof Error ? error.stack || error.message : String(error);
      process.stderr.write(`[rizzup-worker] ${message}\n`);
    }

    if (Date.now() - startedAt >= config.maxRuntimeMs) {
      break;
    }

    await new Promise((resolve) => setTimeout(resolve, config.pollIntervalMs));
  }
}
