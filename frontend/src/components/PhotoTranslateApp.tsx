"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  createTranslationJob,
  deleteJob,
  fetchLanguages,
  fetchSystemStatus,
  getBatchZipUrl,
  getItemDownloadUrl,
  rerenderItem,
  subscribeJobEvents,
} from "../lib/api";
import {
  announceStatus,
  getNotificationPermission,
  playCompletionChime,
  requestBrowserNotificationPermission,
  sendBrowserNotification,
  summarizeJobCompletion,
  type BrowserNotificationPermission,
} from "../lib/notifications";
import type { BlockUpdate, JobInfo, LanguageDTO, SystemStatus } from "../lib/types";
import { AdvancedDrawer } from "./AdvancedDrawer";
import { BatchFilmstrip } from "./BatchFilmstrip";
import { BeforeAfterSlider, type ViewMode } from "./BeforeAfterSlider";
import { CompletionToast, type ToastNotice } from "./CompletionToast";
import {
  AlertCircleIcon,
  ArchiveIcon,
  BellIcon,
  BellOffIcon,
  CheckCircleIcon,
  DownloadIcon,
  PlusIcon,
  SlidersIcon,
  XIcon,
} from "./Icons";
import { LanguageBar } from "./LanguageBar";
import { UploadZone } from "./UploadZone";

export const PhotoTranslateApp: React.FC = () => {
  const [languages, setLanguages] = useState<LanguageDTO[]>([]);
  const [status, setStatus] = useState<SystemStatus | null>(null);

  const [files, setFiles] = useState<File[]>([]);
  const [sourceLang, setSourceLang] = useState("auto");
  const [targetLang, setTargetLang] = useState("am");
  const [translator, setTranslator] = useState("google");

  const [job, setJob] = useState<JobInfo | null>(null);
  const [selectedItemId, setSelectedItemId] = useState<string>("");
  const [viewMode, setViewMode] = useState<ViewMode>("split");
  const [showInspector, setShowInspector] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<string>("");

  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [notifPermission, setNotifPermission] =
    useState<BrowserNotificationPermission>("default");
  const [jobElapsedMs, setJobElapsedMs] = useState<number | null>(null);
  const [showCompletionBanner, setShowCompletionBanner] = useState(true);
  const [toastNotice, setToastNotice] = useState<ToastNotice | null>(null);
  const [hasUnreadCompletion, setHasUnreadCompletion] = useState(false);

  const unsubRef = useRef<(() => void) | null>(null);
  const jobStartedAtRef = useRef<number | null>(null);
  const notifiedJobIdRef = useRef<string | null>(null);
  const alertsEnabledRef = useRef(alertsEnabled);
  alertsEnabledRef.current = alertsEnabled;

  useEffect(() => {
    setNotifPermission(getNotificationPermission());
    try {
      const saved = window.localStorage.getItem("photo-translate:alerts");
      if (saved === "off") setAlertsEnabled(false);
    } catch {
      // ignore storage restrictions
    }
  }, []);

  // Keep browser tab title synced with live batch progress & completion badge
  useEffect(() => {
    if (typeof document === "undefined") return;
    if (!job) {
      document.title = "Photo Translate";
      return;
    }
    if (job.status === "queued" || job.status === "processing") {
      document.title =
        job.total_items > 1
          ? `(${job.completed_items}/${job.total_items}) Translating… — Photo Translate`
          : "Translating… — Photo Translate";
      return;
    }
    if (hasUnreadCompletion) {
      if (job.status === "failed") {
        document.title = "! Failed — Photo Translate";
      } else {
        document.title =
          job.total_items > 1
            ? `✓ Done (${job.total_items}/${job.total_items}) — Photo Translate`
            : "✓ Done — Photo Translate";
      }
      return;
    }
    document.title = "Photo Translate";
  }, [job, hasUnreadCompletion]);

  // Clear unread tab-title completion badge once user returns to the tab
  useEffect(() => {
    if (!hasUnreadCompletion) return;
    const clearUnread = () => {
      if (!document.hidden) {
        setHasUnreadCompletion(false);
      }
    };
    window.addEventListener("focus", clearUnread);
    document.addEventListener("visibilitychange", clearUnread);
    return () => {
      window.removeEventListener("focus", clearUnread);
      document.removeEventListener("visibilitychange", clearUnread);
    };
  }, [hasUnreadCompletion]);

  useEffect(() => {
    Promise.all([fetchLanguages(), fetchSystemStatus()])
      .then(([langs, sys]) => {
        setLanguages(langs);
        setStatus(sys);
        if (langs.length > 0 && !langs.some((l) => l.code === targetLang)) {
          setTargetLang(langs[0].code);
        }
        const firstAvailable = sys.translators.find((t) => t.available);
        if (firstAvailable) {
          setTranslator(firstAvailable.id);
        }
      })
      .catch((err) => {
        setError(
          err instanceof Error
            ? `Unable to reach the local translation server (${err.message}). Make sure the backend is running on port 8000 and refresh.`
            : "Unable to reach the local translation server. Make sure the backend is running on port 8000 and refresh."
        );
      });

    return () => {
      unsubRef.current?.();
    };
  }, []);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT")) {
        return;
      }
      const items = e.clipboardData?.items;
      if (!items) return;
      const pasted: File[] = [];
      for (const item of Array.from(items)) {
        if (item.type.startsWith("image/")) {
          const blob = item.getAsFile();
          if (blob) {
            const ext = item.type.split("/")[1] || "png";
            pasted.push(new File([blob], `pasted-${Date.now()}.${ext}`, { type: item.type }));
          }
        }
      }
      if (pasted.length > 0) {
        setFiles((prev) => [...prev, ...pasted]);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);

  const handleAddFiles = useCallback((incoming: File[]) => {
    setError(null);
    setFiles((prev) => [...prev, ...incoming]);
  }, []);

  const handleRemoveFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleToggleAlerts = async () => {
    if (!alertsEnabled) {
      setAlertsEnabled(true);
      try {
        window.localStorage.setItem("photo-translate:alerts", "on");
      } catch {
        // ignore
      }
      if (getNotificationPermission() === "default") {
        const perm = await requestBrowserNotificationPermission();
        setNotifPermission(perm);
      }
      return;
    }

    if (notifPermission === "default") {
      const perm = await requestBrowserNotificationPermission();
      setNotifPermission(perm);
      return;
    }

    setAlertsEnabled(false);
    try {
      window.localStorage.setItem("photo-translate:alerts", "off");
    } catch {
      // ignore
    }
  };

  const notifyJobFinished = useCallback((finishedJob: JobInfo) => {
    if (notifiedJobIdRef.current === finishedJob.job_id) return;
    notifiedJobIdRef.current = finishedJob.job_id;

    const elapsed =
      jobStartedAtRef.current !== null ? Date.now() - jobStartedAtRef.current : null;
    setJobElapsedMs(elapsed);
    setShowCompletionBanner(true);

    const summary = summarizeJobCompletion(finishedJob, elapsed);
    const announcement = summary.detail
      ? `${summary.headline}. ${summary.detail}.`
      : `${summary.headline}.`;

    announceStatus(announcement, setLiveStatus);
    setToastNotice({
      id: finishedJob.job_id,
      headline: summary.headline,
      detail: summary.detail,
      isFailure: summary.isFailure,
    });

    if (typeof document !== "undefined" && document.hidden) {
      setHasUnreadCompletion(true);
    }

    if (alertsEnabledRef.current) {
      playCompletionChime(summary.isFailure);
      sendBrowserNotification(
        `Photo Translate — ${summary.headline}`,
        summary.detail || "Your translated images are ready to review and save.",
        `photo-translate-${finishedJob.job_id}`
      );
    }
  }, []);

  const handleTranslate = async () => {
    if (files.length === 0) return;
    setError(null);
    setLiveStatus("");
    setToastNotice(null);
    setJobElapsedMs(null);
    setShowCompletionBanner(true);
    setHasUnreadCompletion(false);
    notifiedJobIdRef.current = null;
    jobStartedAtRef.current = Date.now();
    setSubmitting(true);
    unsubRef.current?.();

    // Request browser notification permission inside this direct user click gesture if not yet decided
    if (alertsEnabled && getNotificationPermission() === "default") {
      requestBrowserNotificationPermission()
        .then(setNotifPermission)
        .catch(() => {});
    }

    try {
      const created = await createTranslationJob({
        files,
        sourceLang,
        targetLang,
        translator,
      });
      setJob(created);
      setSelectedItemId(created.items[0]?.item_id || "");
      setShowInspector(false);
      setViewMode("split");

      unsubRef.current = subscribeJobEvents(
        created.job_id,
        (updated) => {
          setJob(updated);
          if (updated.status === "completed" || updated.status === "failed") {
            setSubmitting(false);
            const firstErr = updated.items.find((it) => it.error)?.error;
            if (updated.status === "failed" && firstErr) {
              setError(firstErr);
            }
            notifyJobFinished(updated);
            fetchSystemStatus().then(setStatus).catch(() => {});
          }
        },
        () => {
          setSubmitting(false);
          fetchSystemStatus().then(setStatus).catch(() => {});
        },
        () => {
          setSubmitting(false);
        }
      );
    } catch (err) {
      setSubmitting(false);
      setError(err instanceof Error ? err.message : "Translation failed.");
    }
  };

  const handleStartOver = async () => {
    unsubRef.current?.();
    if (job) {
      deleteJob(job.job_id).catch(() => {});
    }
    setJob(null);
    setSelectedItemId("");
    setFiles([]);
    setShowInspector(false);
    setSubmitting(false);
    setError(null);
    setLiveStatus("");
    setToastNotice(null);
    setJobElapsedMs(null);
    setShowCompletionBanner(true);
    setHasUnreadCompletion(false);
    notifiedJobIdRef.current = null;
    jobStartedAtRef.current = null;
  };

  const handleApplyChanges = async (updates: BlockUpdate[]) => {
    if (!job || !selectedItemId) return;
    setError(null);
    try {
      const updatedItem = await rerenderItem(job.job_id, selectedItemId, updates);
      setJob((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          items: prev.items.map((it) => (it.item_id === updatedItem.item_id ? updatedItem : it)),
        };
      });
      setLiveStatus(`Applied changes to ${updatedItem.filename}.`);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Unable to update the image (${err.message}). Check that the local server is running and try again.`
          : "Unable to update the image. Check that the local server is running and try again."
      );
    }
  };

  const activeItem =
    job?.items.find((it) => it.item_id === selectedItemId) || job?.items[0] || null;
  const progressPct =
    job && job.total_items > 0 ? Math.round((job.completed_items / job.total_items) * 100) : 0;
  const activeWorkersCount = job
    ? job.items.filter(
        (it) =>
          it.status === "detecting" ||
          it.status === "inpainting" ||
          it.status === "translating" ||
          it.status === "rendering"
      ).length
    : 0;
  const completionSummary =
    job && (job.status === "completed" || job.status === "failed")
      ? summarizeJobCompletion(job, jobElapsedMs)
      : null;

  const alertButtonLabel = !alertsEnabled
    ? "Alerts off"
    : notifPermission === "granted"
      ? "Alerts on"
      : notifPermission === "default"
        ? "Notify when done"
        : "Chime on";

  const activeIndex = job && activeItem
    ? job.items.findIndex((it) => it.item_id === activeItem.item_id)
    : -1;

  return (
    <div className="studio-shell">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <header className="studio-topbar">
        <div className="topbar-left">
          <h1 className="wordmark">
            <span className="wordmark-mark" aria-hidden="true" />
            Photo Translate
          </h1>

          {job && (
            <>
              <span className="topbar-divider" aria-hidden="true" />
              <button type="button" className="btn btn-quiet" onClick={handleStartOver}>
                <PlusIcon size={15} /> New upload
              </button>
              {activeItem && (
                <span className="topbar-filemeta" title={activeItem.filename}>
                  {job.total_items > 1 && activeIndex >= 0 && (
                    <span className="topbar-pagecount">
                      {activeIndex + 1} / {job.total_items} ·{" "}
                    </span>
                  )}
                  <bdi>{activeItem.filename}</bdi> ({activeItem.width}×{activeItem.height})
                </span>
              )}
            </>
          )}
        </div>

        <div className="topbar-right">
          {!job ? (
            <LanguageBar
              languages={languages}
              translators={status?.translators || []}
              sourceLang={sourceLang}
              targetLang={targetLang}
              translator={translator}
              onSourceLangChange={setSourceLang}
              onTargetLangChange={setTargetLang}
              onTranslatorChange={setTranslator}
              disabled={submitting}
            />
          ) : (
            activeItem && (
              <>
                <div className="segmented" role="group" aria-label="View mode">
                  <button
                    type="button"
                    className={viewMode === "split" ? "active" : ""}
                    aria-pressed={viewMode === "split"}
                    onClick={() => setViewMode("split")}
                  >
                    Split
                  </button>
                  <button
                    type="button"
                    className={viewMode === "translated" ? "active" : ""}
                    aria-pressed={viewMode === "translated"}
                    onClick={() => setViewMode("translated")}
                  >
                    Translated
                  </button>
                  <button
                    type="button"
                    className={viewMode === "original" ? "active" : ""}
                    aria-pressed={viewMode === "original"}
                    onClick={() => setViewMode("original")}
                  >
                    Original
                  </button>
                </div>

                <button
                  type="button"
                  className={`btn ${showInspector ? "btn-active" : ""}`}
                  aria-expanded={showInspector}
                  aria-controls="text-inspector"
                  onClick={() => setShowInspector((v) => !v)}
                  disabled={activeItem.status !== "completed" && activeItem.status !== "no_text"}
                >
                  <SlidersIcon size={14} /> Edit text
                </button>

                <a
                  href={getItemDownloadUrl(job.job_id, activeItem.item_id)}
                  download={`translated_${activeItem.filename}`}
                  className="btn btn-primary"
                >
                  <DownloadIcon size={14} /> Save image
                </a>

                {job.total_items > 1 && (
                  <a
                    href={getBatchZipUrl(job.job_id)}
                    download={`photo-translate-${job.job_id}.zip`}
                    className="btn"
                  >
                    <ArchiveIcon size={14} /> Save all
                  </a>
                )}
              </>
            )
          )}
        </div>
      </header>

      <main id="main-content" className="studio-main">
        <p className="visually-hidden">Screen reader announcements</p>
        <div className="visually-hidden" aria-live="polite" aria-atomic="true">
          {liveStatus}
        </div>

        {error && !(completionSummary?.isFailure && showCompletionBanner) && (
          <div className="error-strip" role="alert">
            <span className="error-strip-msg">
              <AlertCircleIcon size={16} />
              <span>{error}</span>
            </span>
            <button
              type="button"
              className="btn btn-quiet"
              onClick={() => setError(null)}
              aria-label="Dismiss error"
            >
              <XIcon size={15} />
            </button>
          </div>
        )}

        {!job ? (
          <UploadZone
            files={files}
            onAddFiles={handleAddFiles}
            onRemoveFile={handleRemoveFile}
            onClear={() => setFiles([])}
            onTranslate={handleTranslate}
            submitting={submitting}
          />
        ) : (
          <>
            {(job.status === "queued" || job.status === "processing") && (
              <div className="progress-strip">
                <div className="progress-strip-main">
                  <label htmlFor="batch-progress-bar">
                    Translating {job.completed_items} of {job.total_items}
                    {activeWorkersCount > 1 ? ` · ${activeWorkersCount} active` : ""}
                  </label>
                  <progress
                    id="batch-progress-bar"
                    className="batch-progress"
                    max={Math.max(1, job.total_items)}
                    value={job.completed_items}
                  >
                    {progressPct}%
                  </progress>
                  <span className="progress-pct">{progressPct}%</span>
                </div>
                <button
                  type="button"
                  className={`btn btn-quiet notify-toggle ${alertsEnabled && notifPermission === "granted" ? "is-on" : ""}`}
                  aria-pressed={alertsEnabled}
                  onClick={handleToggleAlerts}
                  title={
                    notifPermission === "denied"
                      ? "Browser notifications are blocked in site settings; sound chime will still play"
                      : alertsEnabled
                        ? "Alert when all images finish translating"
                        : "Turn on completion notifications"
                  }
                >
                  {alertsEnabled ? <BellIcon size={14} /> : <BellOffIcon size={14} />}
                  <span>{alertButtonLabel}</span>
                </button>
              </div>
            )}

            {completionSummary && showCompletionBanner && (
              <div
                className={`progress-strip ${completionSummary.isFailure ? "is-failed" : "is-complete"}`}
                role={completionSummary.isFailure ? "alert" : undefined}
              >
                <div className="progress-strip-main">
                  <span className="completion-badge-icon" aria-hidden="true">
                    {completionSummary.isFailure ? (
                      <AlertCircleIcon size={16} />
                    ) : (
                      <CheckCircleIcon size={16} />
                    )}
                  </span>
                  <strong className="completion-headline">{completionSummary.headline}</strong>
                  {completionSummary.detail && (
                    <span className="completion-detail">{completionSummary.detail}</span>
                  )}
                  {completionSummary.isFailure && error && (
                    <span className="completion-detail">· {error}</span>
                  )}
                </div>
                <button
                  type="button"
                  className="btn btn-quiet btn-icon"
                  onClick={() => {
                    setShowCompletionBanner(false);
                    if (completionSummary.isFailure) setError(null);
                  }}
                  aria-label="Dismiss completion summary"
                >
                  <XIcon size={15} />
                </button>
              </div>
            )}

            {activeItem && (
              <div className={`proofing-body ${showInspector ? "with-inspector" : ""}`}>
                <div className="canvas-column">
                  <BeforeAfterSlider item={activeItem} viewMode={viewMode} />
                  <BatchFilmstrip
                    items={job.items}
                    selectedItemId={activeItem.item_id}
                    onSelect={(id) => setSelectedItemId(id)}
                  />
                </div>

                {showInspector && (
                  <AdvancedDrawer
                    item={activeItem}
                    onApply={handleApplyChanges}
                    onClose={() => setShowInspector(false)}
                  />
                )}
              </div>
            )}
          </>
        )}
      </main>

      <CompletionToast notice={toastNotice} onDismiss={() => setToastNotice(null)} />
    </div>
  );
};
