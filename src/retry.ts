export function calculateRetryDelay(
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number
): number {
  const boundedAttempt = Math.max(1, attempt);
  const exponential = baseDelayMs * 2 ** (boundedAttempt - 1);
  return Math.min(exponential, maxDelayMs);
}
