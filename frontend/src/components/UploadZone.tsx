"use client";

import React, { useRef, useState } from "react";
import { LoaderIcon, UploadIcon, XIcon } from "./Icons";

interface UploadZoneProps {
  files: File[];
  onAddFiles: (incoming: File[]) => void;
  onRemoveFile: (index: number) => void;
  onClear: () => void;
  onTranslate: () => void;
  submitting?: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  files,
  onAddFiles,
  onRemoveFile,
  onClear,
  onTranslate,
  submitting,
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileList = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    onAddFiles(Array.from(list));
  };

  return (
    <section className="empty-desk" aria-label="Image upload">
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp,.zip"
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
          Accepts PNG, JPG, WebP, or a folder archive. Non-text artwork stays untouched.
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
              {files.length} {files.length === 1 ? "file" : "files"} selected
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
                    <LoaderIcon size={15} /> Translating
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
                <span className="file-tag-name">{file.name}</span>
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
