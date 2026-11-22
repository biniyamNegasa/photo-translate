import type { BlockUpdate, JobInfo, JobItemInfo, LanguageDTO, SystemStatus } from "./types";

const API_BASE =
  (typeof import.meta !== "undefined" &&
    (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE) ||
  "http://127.0.0.1:8000";

export function apiUrl(path: string): string {
  if (!path || /^https?:\/\//i.test(path)) return path;
  const base = API_BASE.replace(/\/+$/, "");
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${base}${normalized}`;
}

function hydrateItem(item: JobItemInfo): JobItemInfo {
  return {
    ...item,
    original_url: apiUrl(item.original_url),
    translated_url: apiUrl(item.translated_url),
  };
}

function hydrateJob(job: JobInfo): JobInfo {
  return {
    ...job,
    items: job.items.map(hydrateItem),
  };
}

async function parseError(resp: Response): Promise<string> {
  try {
    const data = await resp.json();
    if (typeof data?.detail === "string") return data.detail;
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
  return hydrateJob(await resp.json());
}

export async function fetchJob(jobId: string): Promise<JobInfo> {
  const resp = await fetch(apiUrl(`/api/jobs/${jobId}`));
  if (!resp.ok) throw new Error(await parseError(resp));
  return hydrateJob(await resp.json());
}

export function subscribeJobEvents(
  jobId: string,
  onUpdate: (job: JobInfo) => void,
  onDone: () => void,
  onError?: (err: Event) => void
): () => void {
  const source = new EventSource(apiUrl(`/api/jobs/${jobId}/events`));

  source.addEventListener("update", (event) => {
    try {
      const parsed: JobInfo = JSON.parse((event as MessageEvent).data);
      onUpdate(hydrateJob(parsed));
    } catch {
      // ignore malformed packet
    }
  });

  source.addEventListener("done", () => {
    onDone();
    source.close();
  });

  source.onerror = (err) => {
    if (onError) onError(err);
  };

  return () => {
    source.close();
  };
}

export async function rerenderItem(
  jobId: string,
  itemId: string,
  blocks: BlockUpdate[]
): Promise<JobItemInfo> {
  const resp = await fetch(apiUrl(`/api/jobs/${jobId}/items/${itemId}/rerender`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blocks }),
  });
  if (!resp.ok) throw new Error(await parseError(resp));
  return hydrateItem(await resp.json());
}

export async function deleteJob(jobId: string): Promise<void> {
  await fetch(apiUrl(`/api/jobs/${jobId}`), { method: "DELETE" });
}

export function getItemDownloadUrl(jobId: string, itemId: string): string {
  return apiUrl(`/api/jobs/${jobId}/items/${itemId}/translated?download=true`);
}

export function getBatchZipUrl(jobId: string): string {
  return apiUrl(`/api/jobs/${jobId}/download`);
}
