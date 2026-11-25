"use client";

import React, { useRef, useState } from "react";
import { formatCount } from "../lib/notifications";
import { LoaderIcon, UploadIcon, XIcon } from "./Icons";

export const DEFAULT_MAX_UPLOAD_FILES = 100;
export const DEFAULT_MAX_UPLOAD_MB = 100;

const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".zip"]);
const ALLOWED_IMAGE_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/bmp",
]);

function getFileExtension(filename: string): string {
  const dotIdx = filename.lastIndexOf(".");
  if (dotIdx <= 0) return "";
  return filename.slice(dotIdx).toLowerCase();
}

function isAllowedUploadFile(file: File): boolean {
  const ext = getFileExtension(file.name);
  if (!ALLOWED_EXTENSIONS.has(ext)) return false;

  const mime = (file.type || "").toLowerCase();
  if (mime === "image/svg+xml") return false;
  if (ext !== ".zip" && mime && mime.startsWith("image/") && !ALLOWED_IMAGE_MIMES.has(mime)) {
    return false;
  }
  return true;
}

export interface UploadValidationResult {
  accepted: File[];
  error: string | null;
}

export function validateUploadFiles(
  existing: File[],
  incoming: File[],
  options?: { maxFiles?: number; maxUploadMb?: number }
): UploadValidationResult {
  const maxFiles = Math.max(1, options?.maxFiles ?? DEFAULT_MAX_UPLOAD_FILES);
  const maxUploadMb = Math.max(1, options?.maxUploadMb ?? DEFAULT_MAX_UPLOAD_MB);
  const maxTotalBytes = maxUploadMb * 1024 * 1024;

  const accepted: File[] = [];
  let runningBytes = existing.reduce((sum, f) => sum + f.size, 0);
  let unsupportedCount = 0;
  let emptyCount = 0;
  let countLimitHit = false;
  let sizeLimitHit = false;

  for (const file of incoming) {
    if (!isAllowedUploadFile(file)) {
      unsupportedCount += 1;
      continue;
    }
    if (file.size <= 0) {
      emptyCount += 1;
      continue;
    }
    if (existing.length + accepted.length >= maxFiles) {
      countLimitHit = true;
      continue;
    }
    if (runningBytes + file.size > maxTotalBytes) {
      sizeLimitHit = true;
      continue;
    }
    accepted.push(file);
    runningBytes += file.size;
  }

  const reasons: string[] = [];
  if (unsupportedCount > 0) {
    reasons.push(
      `Skipped ${formatCount(unsupportedCount, "unsupported file", "unsupported files")} (allowed: PNG, JPG, WebP, BMP, ZIP).`
    );
  }
  if (emptyCount > 0) {
    reasons.push(`Skipped ${formatCount(emptyCount, "empty file", "empty files")}.`);
  }
  if (countLimitHit) {
    reasons.push(`Maximum ${maxFiles} files per batch.`);
  }
  if (sizeLimitHit) {
    reasons.push(`Total upload size cannot exceed ${maxUploadMb} MB.`);
  }

  return {
    accepted,
    error: reasons.length > 0 ? reasons.join(" ") : null,
  };
}

interface UploadZoneProps {
  files: File[];
  onAddFiles: (incoming: File[]) => void;
  onRemoveFile: (index: number) => void;
  onClear: () => void;
  onTranslate: () => void;
  onValidationError?: (message: string) => void;
  maxFiles?: number;
  maxUploadMb?: number;
  submitting?: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  files,
  onAddFiles,
  onRemoveFile,
  onClear,
  onTranslate,
  onValidationError,
  maxFiles = DEFAULT_MAX_UPLOAD_FILES,
  maxUploadMb = DEFAULT_MAX_UPLOAD_MB,
  submitting,
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileList = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    const { accepted, error } = validateUploadFiles(files, Array.from(list), {
      maxFiles,
      maxUploadMb,
    });
    if (accepted.length > 0) {
      onAddFiles(accepted);
    }
    if (error) {
      onValidationError?.(error);
    }
  };

  return (
    <section className="empty-desk" aria-label="Image upload">
      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.webp,.bmp,.zip,image/png,image/jpeg,image/webp,image/bmp,application/zip"
        multiple
        hidden
        onChange={(e) => {
          handleFileList(e.target.files);
          e.target.value = "";
        }}
      />

      <button
        type="button"
        className={`drop-sheet ${dragActive ? "drag-active" : ""}`}
        onClick={() => !submitting && inputRef.current?.click()}
        disabled={submitting}
        onDragOver={(e) => {
          e.preventDefault();
          if (!submitting) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          if (!submitting) handleFileList(e.dataTransfer.files);
        }}
      >
        <span className="drop-title">Drop an image or a ZIP of pages</span>
        <span className="drop-copy">
          Accepts PNG, JPG, WebP, BMP, or a ZIP archive (up to {maxFiles} files, {maxUploadMb} MB
          total). Non-text artwork stays untouched.
        </span>
        <span className="drop-actions-row">
          <span className="drop-cta">
            <UploadIcon size={15} /> Choose files
          </span>
          <span className="drop-shortcut">
            or paste with <kbd>Ctrl+V</kbd>
          </span>
        </span>
      </button>

      {files.length > 0 && (
        <div className="queue-tray">
          <div className="queue-tray-head">
            <span className="queue-count">
              {formatCount(files.length, "file", "files")} selected
            </span>
            <div className="queue-actions">
              <button
                type="button"
                className="btn btn-quiet"
                onClick={onClear}
                disabled={submitting}
              >
                Clear
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={onTranslate}
                disabled={submitting}
              >
                {submitting ? (
                  <>
                    <LoaderIcon size={15} /> Translating…
                  </>
                ) : (
                  "Translate"
                )}
              </button>
            </div>
          </div>

          <ul className="file-list" role="list" aria-label="Selected files">
            {files.slice(0, 12).map((file, idx) => (
              <li key={`${file.name}-${idx}`} className="file-tag">
                <bdi className="file-tag-name" title={file.name}>
                  {file.name}
                </bdi>
                <button
                  type="button"
                  className="tag-remove-btn"
                  aria-label={`Remove ${file.name}`}
                  onClick={() => onRemoveFile(idx)}
                  disabled={submitting}
                >
                  <XIcon size={13} />
                </button>
              </li>
            ))}
            {files.length > 12 && (
              <li className="file-tag">
                <span className="file-tag-name">+{files.length - 12} more</span>
              </li>
            )}
          </ul>
        </div>
      )}
    </section>
  );
};
