import type { BlockUpdate, JobInfo, JobItemInfo, LanguageDTO, SystemStatus } from "./types";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const JOB_ID_RE = /^[a-f0-9]{12,64}$/;
const ITEM_ID_RE = /^img_[0-9]{1,6}$/;
const VALID_JOB_STATUSES = new Set(["queued", "processing", "completed", "failed", "partial"]);
const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "partial"]);
const MAX_SSE_ERRORS = 5;
const MAX_ERROR_LENGTH = 240;
const MAX_FILENAME_LENGTH = 120;

function resolveApiBase(): { base: string; origin: string; basePath: string } {
  const raw =
    typeof import.meta !== "undefined"
      ? (import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE
      : undefined;

  if (typeof raw === "string" && raw.trim().length > 0) {
    try {
      const parsed = new URL(raw.trim());
      if (
        (parsed.protocol === "http:" || parsed.protocol === "https:") &&
        !parsed.username &&
        !parsed.password
      ) {
        const basePath = parsed.pathname.replace(/\/+$/, "");
        return {
          base: `${parsed.origin}${basePath}`,
          origin: parsed.origin,
          basePath,
        };
      }
    } catch {
      // Fall back to safe localhost default
    }
  }

  const fallback = new URL(DEFAULT_API_BASE);
  return {
    base: fallback.origin,
    origin: fallback.origin,
    basePath: "",
  };
}

const API_CONFIG = resolveApiBase();
export const API_BASE = API_CONFIG.base;
const API_BASE_ORIGIN = API_CONFIG.origin;
const API_BASE_PATH = API_CONFIG.basePath;

function assertValidJobId(jobId: string): string {
  if (typeof jobId !== "string" || !JOB_ID_RE.test(jobId)) {
    throw new Error("Invalid job identifier.");
  }
  return encodeURIComponent(jobId);
}

function assertValidItemId(itemId: string): string {
  if (typeof itemId !== "string" || !ITEM_ID_RE.test(itemId)) {
    throw new Error("Invalid item identifier.");
  }
  return encodeURIComponent(itemId);
}

function resolveScopedApiUrl(path: string, requiredPrefix: string): string | null {
  if (typeof path !== "string") return null;
  const trimmed = path.trim();
  if (!trimmed || trimmed.startsWith("//") || /[\\\x00-\x1f\x7f]/.test(trimmed)) {
    return null;
  }

  const expectedPrefix = `${API_BASE_PATH}${requiredPrefix}`;

  try {
    let resolved: URL;
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) {
      resolved = new URL(trimmed);
    } else {
      const normalized = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
      const withBase =
        API_BASE_PATH &&
        normalized !== API_BASE_PATH &&
        !normalized.startsWith(`${API_BASE_PATH}/`)
          ? `${API_BASE_PATH}${normalized}`
          : normalized;
      resolved = new URL(withBase, API_BASE_ORIGIN);
    }

    if (resolved.protocol !== "http:" && resolved.protocol !== "https:") {
      return null;
    }
    if (resolved.origin !== API_BASE_ORIGIN) {
      return null;
    }
    if (!resolved.pathname.startsWith(expectedPrefix)) {
      return null;
    }
    return resolved.toString();
  } catch {
    return null;
  }
}

export function apiUrl(path: string): string {
  const resolved = resolveScopedApiUrl(path, "/api/");
  if (!resolved) {
    throw new Error("Blocked request to an untrusted API URL.");
  }
  return resolved;
}

export function safeMediaUrl(path: string): string {
  return resolveScopedApiUrl(path, "/api/jobs/") ?? "";
}

export function sanitizeErrorMessage(message: string, fallback = "Unexpected error."): string {
  if (typeof message !== "string") return fallback;
  const cleaned = message
    .replace(/[\x00-\x1f\x7f\u202a-\u202e\u2066-\u2069]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) return fallback;
  if (cleaned.length <= MAX_ERROR_LENGTH) return cleaned;
  return `${cleaned.slice(0, MAX_ERROR_LENGTH - 1)}…`;
}

export function sanitizeDownloadName(name: string, fallback = "translated-image.png"): string {
  if (typeof name !== "string") return fallback;
  const stripped = name
    .replace(/[\x00-\x1f\x7f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g, "")
    .replace(/[/\\:*?"<>|]+/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[.\s]+/, "");

  if (!stripped) return fallback;
  if (stripped.length <= MAX_FILENAME_LENGTH) return stripped;

  const dotIdx = stripped.lastIndexOf(".");
  if (dotIdx > 0 && stripped.length - dotIdx <= 10) {
    const ext = stripped.slice(dotIdx);
    const stem = stripped.slice(0, MAX_FILENAME_LENGTH - ext.length).replace(/[.\s]+$/, "");
    return stem ? `${stem}${ext}` : fallback;
  }
  return stripped.slice(0, MAX_FILENAME_LENGTH);
}

function hydrateItem(item: JobItemInfo): JobItemInfo {
  return {
    ...item,
    original_url: item.original_url ? safeMediaUrl(item.original_url) : "",
    translated_url: item.translated_url ? safeMediaUrl(item.translated_url) : "",
  };
}

function hydrateJob(job: JobInfo): JobInfo {
  return {
    ...job,
    items: job.items.map(hydrateItem),
  };
}

function isValidJobItem(value: unknown): value is JobItemInfo {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return (
    typeof item.item_id === "string" &&
    ITEM_ID_RE.test(item.item_id) &&
    typeof item.filename === "string" &&
    typeof item.status === "string" &&
    typeof item.original_url === "string" &&
    typeof item.translated_url === "string" &&
    Array.isArray(item.blocks)
  );
}

export function isValidJobInfo(data: unknown): data is JobInfo {
  if (!data || typeof data !== "object") return false;
  const job = data as Record<string, unknown>;
  return (
    typeof job.job_id === "string" &&
    JOB_ID_RE.test(job.job_id) &&
    typeof job.status === "string" &&
    VALID_JOB_STATUSES.has(job.status) &&
    typeof job.total_items === "number" &&
    typeof job.completed_items === "number" &&
    Array.isArray(job.items) &&
    job.items.every(isValidJobItem)
  );
}

async function parseError(resp: Response): Promise<string> {
  try {
    const data = await resp.json();
    if (typeof data?.detail === "string" && data.detail.trim()) {
      return sanitizeErrorMessage(data.detail, `Request failed (${resp.status})`);
    }
  } catch {
    // ignore JSON error
  }
  return `Request failed (${resp.status})`;
}

export async function fetchLanguages(): Promise<LanguageDTO[]> {
  const resp = await fetch(apiUrl("/api/languages"));
  if (!resp.ok) throw new Error(await parseError(resp));
  return resp.json();
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const resp = await fetch(apiUrl("/api/status"));
  if (!resp.ok) throw new Error(await parseError(resp));
  return resp.json();
}

export async function createTranslationJob(params: {
  files: File[];
  targetLang: string;
  sourceLang: string;
  translator: string;
}): Promise<JobInfo> {
  const form = new FormData();
  for (const file of params.files) {
    form.append("files", file);
  }
  form.append("target_lang", params.targetLang);
  form.append("source_lang", params.sourceLang);
  form.append("translator", params.translator);

  const resp = await fetch(apiUrl("/api/jobs"), {
    method: "POST",
    body: form,
  });
  if (!resp.ok) throw new Error(await parseError(resp));
  const payload = await resp.json();
  if (!isValidJobInfo(payload)) {
    throw new Error("Received an unexpected job response from the server.");
  }
  return hydrateJob(payload);
}

export async function fetchJob(jobId: string): Promise<JobInfo> {
  const safeJobId = assertValidJobId(jobId);
  const resp = await fetch(apiUrl(`/api/jobs/${safeJobId}`));
  if (!resp.ok) throw new Error(await parseError(resp));
  const payload = await resp.json();
  if (!isValidJobInfo(payload)) {
    throw new Error("Received an unexpected job response from the server.");
  }
  return hydrateJob(payload);
}

export function subscribeJobEvents(
  jobId: string,
  onUpdate: (job: JobInfo) => void,
  onDone: () => void,
  onError?: (err: Error) => void
): () => void {
  const safeJobId = assertValidJobId(jobId);
  const source = new EventSource(apiUrl(`/api/jobs/${safeJobId}/events`));
  let closed = false;
  let consecutiveErrors = 0;

  const close = () => {
    if (closed) return;
    closed = true;
    source.close();
  };

  source.addEventListener("update", (event) => {
    if (closed) return;
    try {
      const parsed: unknown = JSON.parse((event as MessageEvent).data);
      if (!isValidJobInfo(parsed)) {
        return;
      }
      consecutiveErrors = 0;
      const hydrated = hydrateJob(parsed);
      onUpdate(hydrated);
      if (TERMINAL_JOB_STATUSES.has(hydrated.status)) {
        close();
      }
    } catch {
      // ignore malformed packet
    }
  });

  source.addEventListener("done", () => {
    if (closed) return;
    close();
    onDone();
  });

  source.onerror = () => {
    if (closed) return;
    consecutiveErrors += 1;
    if (source.readyState === EventSource.CLOSED || consecutiveErrors >= MAX_SSE_ERRORS) {
      close();
      if (onError) {
        onError(new Error("Lost connection to the translation progress stream."));
      }
    }
  };

  return close;
}

export async function rerenderItem(
  jobId: string,
  itemId: string,
  blocks: BlockUpdate[]
): Promise<JobItemInfo> {
  const safeJobId = assertValidJobId(jobId);
  const safeItemId = assertValidItemId(itemId);
  const resp = await fetch(apiUrl(`/api/jobs/${safeJobId}/items/${safeItemId}/rerender`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blocks }),
  });
  if (!resp.ok) throw new Error(await parseError(resp));
  const payload = await resp.json();
  if (!isValidJobItem(payload)) {
    throw new Error("Received an unexpected item response from the server.");
  }
  return hydrateItem(payload);
}

export async function deleteJob(jobId: string): Promise<void> {
  const safeJobId = assertValidJobId(jobId);
  const resp = await fetch(apiUrl(`/api/jobs/${safeJobId}`), { method: "DELETE" });
  if (resp.status === 404) return;
  if (!resp.ok) {
    throw new Error(await parseError(resp));
  }
}

export function getItemDownloadUrl(jobId: string, itemId: string): string {
  const safeJobId = assertValidJobId(jobId);
  const safeItemId = assertValidItemId(itemId);
  return safeMediaUrl(`/api/jobs/${safeJobId}/items/${safeItemId}/translated?download=true`);
}

export function getBatchZipUrl(jobId: string): string {
  const safeJobId = assertValidJobId(jobId);
  return safeMediaUrl(`/api/jobs/${safeJobId}/download`);
}

export async function downloadMedia(url: string, filename: string): Promise<void> {
  const safeUrl = safeMediaUrl(url);
  if (!safeUrl) {
    throw new Error("Blocked download from an untrusted URL.");
  }

  const resp = await fetch(safeUrl, { credentials: "omit" });
  if (!resp.ok) {
    throw new Error(await parseError(resp));
  }

  const blob = await resp.blob();
  const objectUrl = URL.createObjectURL(blob);
  const safeName = sanitizeDownloadName(filename);

  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = safeName;
  link.rel = "noopener noreferrer";
  link.style.display = "none";
  document.body.appendChild(link);
  try {
    link.click();
  } finally {
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  }
}
