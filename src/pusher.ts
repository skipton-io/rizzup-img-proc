import Pusher from "pusher";
import { JobStatusEventPayload, PusherConfig, WorkerConfig } from "./types";

type PusherClientLike = {
  trigger(channel: string, eventName: string, payload: JobStatusEventPayload): Promise<unknown>;
};

type PusherClientFactory = (config: PusherConfig) => PusherClientLike;

const STATUS_EVENT_NAME = "status";

function defaultClientFactory(config: PusherConfig): PusherClientLike {
  return new Pusher({
    appId: config.appId,
    key: config.key,
    secret: config.secret,
    cluster: config.cluster,
    useTLS: true
  });
}

let cachedClient: PusherClientLike | null = null;
let cachedConfigKey = "";
let clientFactory: PusherClientFactory = defaultClientFactory;

function cacheKeyForConfig(config: PusherConfig): string {
  return [config.appId, config.key, config.secret, config.cluster].join("\n");
}

function getClient(config: WorkerConfig): PusherClientLike | null {
  if (!config.pusher) {
    return null;
  }

  const nextConfigKey = cacheKeyForConfig(config.pusher);
  if (cachedClient && cachedConfigKey === nextConfigKey) {
    return cachedClient;
  }

  try {
    cachedClient = clientFactory(config.pusher);
    cachedConfigKey = nextConfigKey;
    return cachedClient;
  } catch (error) {
    cachedClient = null;
    cachedConfigKey = "";
    console.warn("[pusher] failed to create client", {
      message: error instanceof Error ? error.message : String(error)
    });
    return null;
  }
}

export function channelForQueueKey(queueKey: string): string {
  const raw = String(queueKey || "").trim();
  if (!raw) {
    return "";
  }

  const safe = raw.replace(/[^A-Za-z0-9_\-=@,.;]/g, "_").slice(0, 150);
  return `job-${safe}`;
}

export async function publishJobStatus(
  config: WorkerConfig,
  queueKey: string,
  payload: JobStatusEventPayload
): Promise<boolean> {
  const channel = channelForQueueKey(queueKey);
  if (!channel) {
    return false;
  }

  const client = getClient(config);
  if (!client) {
    return false;
  }

  try {
    await client.trigger(channel, STATUS_EVENT_NAME, payload);
    return true;
  } catch (error) {
    console.warn("[pusher] publishJobStatus failed", {
      channel,
      message: error instanceof Error ? error.message : String(error)
    });
    return false;
  }
}

export function setPusherClientFactoryForTests(factory: PusherClientFactory | null): void {
  clientFactory = factory ?? defaultClientFactory;
  cachedClient = null;
  cachedConfigKey = "";
}

export function resetPusherClientForTests(): void {
  setPusherClientFactoryForTests(null);
}
