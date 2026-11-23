import type { JobInfo } from "./types";

export type BrowserNotificationPermission = NotificationPermission | "unsupported";

export interface JobCompletionSummary {
  headline: string;
  detail: string;
  isFailure: boolean;
  translatedCount: number;
  noTextCount: number;
  failedCount: number;
}

export function getNotificationPermission(): BrowserNotificationPermission {
  if (typeof window === "undefined" || !("Notification" in window)) {
    return "unsupported";
  }
  return window.Notification.permission;
}

export async function requestBrowserNotificationPermission(): Promise<BrowserNotificationPermission> {
  if (typeof window === "undefined" || !("Notification" in window)) {
    return "unsupported";
  }
  if (window.Notification.permission !== "default") {
    return window.Notification.permission;
  }
  try {
    return await window.Notification.requestPermission();
  } catch {
    return window.Notification.permission;
  }
}

export function sendBrowserNotification(title: string, body: string, tag: string): void {
  if (typeof window === "undefined" || !("Notification" in window)) return;
  if (window.Notification.permission !== "granted") return;

  try {
    const notification = new window.Notification(title, {
      body,
      tag,
    });
    notification.onclick = () => {
      window.focus();
      notification.close();
    };
  } catch {
    // Ignore browser notification errors in restricted contexts
  }
}

export function playCompletionChime(isFailure = false): void {
  if (typeof window === "undefined") return;
  const AudioCtx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioCtx) return;

  try {
    const ctx = new AudioCtx();
    const now = ctx.currentTime;
    const notes = isFailure ? [330, 262] : [587.33, 880]; // D5 -> A5 pleasant bell, or E4 -> C4

    notes.forEach((freq, idx) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, now + idx * 0.11);

      gain.gain.setValueAtTime(0.0001, now + idx * 0.11);
      gain.gain.exponentialRampToValueAtTime(0.08, now + idx * 0.11 + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.11 + 0.28);

      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now + idx * 0.11);
      osc.stop(now + idx * 0.11 + 0.3);
    });

    window.setTimeout(() => {
      ctx.close().catch(() => {});
    }, 650);
  } catch {
    // Ignore audio context autoplay restrictions
  }
}

export function announceStatus(
  message: string,
  setLiveFallback: (msg: string) => void
): void {
  if (!message) return;
  const doc = typeof document !== "undefined" ? (document as Document & { ariaNotify?: (msg: string) => void }) : null;
  if (doc && typeof doc.ariaNotify === "function") {
    try {
      doc.ariaNotify(message);
      return;
    } catch {
      // Fall through to ARIA live region fallback
    }
  }
  setLiveFallback(message);
}

export function formatElapsed(ms: number | null): string {
  if (ms === null || ms < 500) return "";
  const totalSec = Math.max(1, Math.round(ms / 1000));
  if (totalSec < 60) return `${totalSec}s`;
  const mins = Math.floor(totalSec / 60);
  const secs = totalSec % 60;
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

const pluralRules = typeof Intl !== "undefined" ? new Intl.PluralRules("en") : null;
const numberFmt = typeof Intl !== "undefined" ? new Intl.NumberFormat() : null;

export function formatCount(count: number, singular: string, plural: string): string {
  const formatted = numberFmt ? numberFmt.format(count) : String(count);
  const category = pluralRules ? pluralRules.select(count) : count === 1 ? "one" : "other";
  return `${formatted} ${category === "one" ? singular : plural}`;
}

export function summarizeJobCompletion(job: JobInfo, elapsedMs: number | null): JobCompletionSummary {
  const total = job.total_items;
  const translatedCount = job.items.filter((it) => it.status === "completed").length;
  const noTextCount = job.items.filter((it) => it.status === "no_text").length;
  const failedCount = job.items.filter((it) => it.status === "failed").length;
  const elapsedLabel = formatElapsed(elapsedMs);
  const isFailure = job.status === "failed";

  let headline: string;
  if (isFailure) {
    headline = total === 1 ? "Translation failed" : `Translation failed (${failedCount} of ${total})`;
  } else if (total === 1) {
    headline =
      noTextCount === 1
        ? "Finished — no text found"
        : "Translation complete";
  } else if (failedCount > 0) {
    headline = `Batch finished — ${translatedCount + noTextCount} of ${total} succeeded`;
  } else {
    headline = `All ${formatCount(total, "image", "images")} finished`;
  }

  const parts: string[] = [];
  if (total === 1 && translatedCount === 1) {
    const blocks = job.items[0]?.blocks?.length ?? 0;
    if (blocks > 0) {
      parts.push(formatCount(blocks, "text block", "text blocks"));
    }
  } else if (total > 1) {
    if (translatedCount > 0) parts.push(`${translatedCount} translated`);
    if (noTextCount > 0) parts.push(`${noTextCount} no text`);
    if (failedCount > 0) parts.push(`${failedCount} failed`);
  }
  if (elapsedLabel) {
    parts.push(`in ${elapsedLabel}`);
  }

  return {
    headline,
    detail: parts.join(" · "),
    isFailure,
    translatedCount,
    noTextCount,
    failedCount,
  };
}
