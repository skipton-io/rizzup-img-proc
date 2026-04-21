import test from "node:test";
import assert from "node:assert/strict";
import {
  channelForQueueKey,
  publishJobStatus,
  resetPusherClientForTests,
  setPusherClientFactoryForTests
} from "../src/pusher";
import { WorkerConfig } from "../src/types";

function baseConfig(): WorkerConfig {
  return {
    netlifySiteId: "site",
    netlifyAccessToken: "token",
    queueStore: "queue",
    statusStore: "status",
    resultsStore: "results",
    assetsStore: "assets",
    locksStore: "locks",
    deadLetterStore: "dead",
    maxRuntimeMs: 55_000,
    pollIntervalMs: 1,
    maxJobsPerPoll: 1,
    lockTtlMs: 60_000,
    maxAttempts: 3,
    retryBaseDelayMs: 1_000,
    retryMaxDelayMs: 5_000,
    workerId: "worker_test",
    previewWatermarkText: "RizzUp Preview",
    previewWatermarkLogoPath: undefined,
    resultsDir: "artifacts",
    archiveBackend: "local",
    imageArchiveRoot: "archive",
    sourceImageRoot: undefined,
    localRenderRoot: "renders",
    sftpHost: undefined,
    sftpPort: 22,
    sftpUsername: undefined,
    sftpPassword: undefined,
    sftpStrictHostKey: false,
    sftpHostKey: undefined,
    pythonExecutable: "python3",
    pythonScript: "scripts/gpu_pipeline.py",
    faceCascadePath: undefined,
    eyeCascadePath: undefined,
    fireRedEnabled: false,
    fireRedModelId: "Tongyi-MAI/Z-Image-Turbo",
    fireRedPrompt: "Beautify this image",
    fireRedInferenceSteps: 30,
    fireRedTrueCfgScale: 4,
    analysisMaxSize: 100,
    previewMaxSize: 512,
    finalDecisionMaxSize: 512,
    finalMinWidth: 1024,
    finalMinHeight: 1280
  };
}

test.afterEach(() => {
  resetPusherClientForTests();
});

test("channelForQueueKey returns an empty string for blank queue keys", () => {
  assert.equal(channelForQueueKey("   "), "");
});

test("channelForQueueKey sanitizes unsupported characters and truncates long keys", () => {
  const longQueueKey = `${"a/".repeat(100)}tail`;
  const channel = channelForQueueKey(longQueueKey);

  assert.ok(channel.startsWith("job-"));
  assert.ok(channel.length <= 154);
  assert.ok(!channel.includes("/"));
  assert.match(channel, /^job-[A-Za-z0-9_\-=@,.;]+$/);
});

test("publishJobStatus returns false when pusher is not configured", async () => {
  const published = await publishJobStatus(baseConfig(), "generate_preview/job.json", {
    status: "processing",
    attempts: 1
  });

  assert.equal(published, false);
});

test("publishJobStatus returns false when the client factory throws", async () => {
  setPusherClientFactoryForTests(() => {
    throw new Error("factory failed");
  });

  const published = await publishJobStatus(
    {
      ...baseConfig(),
      pusher: {
        appId: "app",
        key: "key",
        secret: "secret",
        cluster: "eu"
      }
    },
    "generate_preview/job.json",
    {
      status: "processing",
      attempts: 1
    }
  );

  assert.equal(published, false);
});

test("publishJobStatus returns false when the client trigger rejects", async () => {
  setPusherClientFactoryForTests(() => ({
    async trigger() {
      throw new Error("trigger failed");
    }
  }));

  const published = await publishJobStatus(
    {
      ...baseConfig(),
      pusher: {
        appId: "app",
        key: "key",
        secret: "secret",
        cluster: "eu"
      }
    },
    "generate_preview/job.json",
    {
      status: "processing",
      attempts: 1
    }
  );

  assert.equal(published, false);
});

test("publishJobStatus reuses the cached client for repeated calls with the same config", async () => {
  let factoryCalls = 0;
  const triggerCalls: string[] = [];
  setPusherClientFactoryForTests(() => {
    factoryCalls += 1;
    return {
      async trigger(channel: string) {
        triggerCalls.push(channel);
      }
    };
  });

  const config = {
    ...baseConfig(),
    pusher: {
      appId: "app",
      key: "key",
      secret: "secret",
      cluster: "eu"
    }
  };

  const first = await publishJobStatus(config, "generate_preview/job_one.json", {
    status: "processing",
    attempts: 1
  });
  const second = await publishJobStatus(config, "generate_preview/job_two.json", {
    status: "completed",
    attempts: 2
  });

  assert.equal(first, true);
  assert.equal(second, true);
  assert.equal(factoryCalls, 1);
  assert.equal(triggerCalls.length, 2);
});
