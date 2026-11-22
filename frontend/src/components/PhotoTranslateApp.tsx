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
import type { BlockUpdate, JobInfo, LanguageDTO, SystemStatus } from "../lib/types";
import { AdvancedDrawer } from "./AdvancedDrawer";
import { BatchFilmstrip } from "./BatchFilmstrip";
import { BeforeAfterSlider, type ViewMode } from "./BeforeAfterSlider";
import {
  AlertCircleIcon,
  ArchiveIcon,
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

  const unsubRef = useRef<(() => void) | null>(null);

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
            ? `Could not connect to the translation service: ${err.message}`
            : "Could not connect to the translation service."
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

  const handleTranslate = async () => {
    if (files.length === 0) return;
    setError(null);
    setLiveStatus("");
    setSubmitting(true);
    unsubRef.current?.();

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
            setLiveStatus(
              updated.status === "completed"
                ? `Translated ${updated.total_items} ${updated.total_items === 1 ? "image" : "images"}.`
                : "Translation failed."
            );
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
      setError(err instanceof Error ? err.message : "Could not re-render image.");
    }
  };

  const activeItem =
    job?.items.find((it) => it.item_id === selectedItemId) || job?.items[0] || null;
  const progressPct =
    job && job.total_items > 0 ? Math.round((job.completed_items / job.total_items) * 100) : 0;

  return (
    <div className="studio-shell">
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
                  {activeItem.filename} ({activeItem.width}×{activeItem.height})
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
        <div className="visually-hidden" aria-live="polite" aria-atomic="true">
          {liveStatus}
        </div>

        {error && (
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
                <label htmlFor="batch-progress-bar">
                  Translating {job.completed_items} of {job.total_items}
                </label>
                <progress
                  id="batch-progress-bar"
                  className="batch-progress"
                  max={Math.max(1, job.total_items)}
                  value={job.completed_items}
                >
                  {progressPct}%
                </progress>
                <span>{progressPct}%</span>
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
    </div>
  );
};
