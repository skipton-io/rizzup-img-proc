import { getStore } from "@netlify/blobs";
import { WorkerConfig, WorkerStores } from "./types";

export function createStores(config: WorkerConfig): WorkerStores {
  const base = {
    siteID: config.netlifySiteId,
    token: config.netlifyAccessToken
  };

  return {
    queue: getStore(config.queueStore, base),
    status: getStore(config.statusStore, base),
    results: getStore(config.resultsStore, base),
    locks: getStore(config.locksStore, base),
    deadLetter: getStore(config.deadLetterStore, base)
  };
}
