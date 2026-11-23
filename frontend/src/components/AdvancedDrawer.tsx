"use client";

import React, { useEffect, useId, useState } from "react";
import type { BlockUpdate, JobItemInfo } from "../lib/types";
import { LoaderIcon, XIcon } from "./Icons";

interface AdvancedDrawerProps {
  item: JobItemInfo;
  onApply: (updates: BlockUpdate[]) => Promise<void>;
  onClose: () => void;
}

interface DraftBlock {
  id: string;
  source_text: string;
  translated_text: string;
  enabled: boolean;
}

export const AdvancedDrawer: React.FC<AdvancedDrawerProps> = ({ item, onApply, onClose }) => {
  const idPrefix = useId();
  const [drafts, setDrafts] = useState<DraftBlock[]>([]);
  const [applying, setApplying] = useState(false);

  useEffect(() => {
    setDrafts(
      item.blocks.map((b) => ({
        id: b.id,
        source_text: b.source_text,
        translated_text: b.translated_text,
        enabled: b.enabled,
      }))
    );
  }, [item.item_id, item.blocks]);

  const handleApply = async () => {
    setApplying(true);
    try {
      await onApply(
        drafts.map((d) => ({
          id: d.id,
          translated_text: d.translated_text,
          enabled: d.enabled,
        }))
      );
    } finally {
      setApplying(false);
    }
  };

  return (
    <aside id="text-inspector" className="inspector-panel" aria-label="Text blocks">
      <div className="inspector-header">
        <div>
          <h2>Text blocks</h2>
          <p className="inspector-sub">
            Uncheck a block to keep the original pixels, or edit the translation below.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-quiet btn-icon"
          onClick={onClose}
          aria-label="Close text blocks panel"
        >
          <XIcon size={17} />
        </button>
      </div>

      <div className="inspector-body">
        {drafts.length === 0 ? (
          <p className="inspector-sub">
            No text detected in this image. Original artwork is preserved untouched.
          </p>
        ) : (
          drafts.map((blk, idx) => {
            const checkId = `${idPrefix}-chk-${blk.id}`;
            const textId = `${idPrefix}-txt-${blk.id}`;
            const srcId = `${idPrefix}-src-${blk.id}`;
            return (
              <div key={blk.id} className={`block-item ${blk.enabled ? "" : "disabled"}`}>
                <div className="block-row">
                  <label htmlFor={checkId} className="block-toggle">
                    <input
                      id={checkId}
                      type="checkbox"
                      className="brand-checkbox"
                      checked={blk.enabled}
                      onChange={(e) => {
                        const next = [...drafts];
                        next[idx] = { ...next[idx], enabled: e.target.checked };
                        setDrafts(next);
                      }}
                    />
                    <span>Block {idx + 1}</span>
                  </label>
                </div>

                <div id={srcId} className="block-source">
                  <bdi>{blk.source_text}</bdi>
                </div>

                <label htmlFor={textId} className="visually-hidden">
                  Translation for block {idx + 1}
                </label>
                <textarea
                  id={textId}
                  aria-describedby={srcId}
                  className="block-textarea"
                  value={blk.translated_text}
                  disabled={!blk.enabled}
                  onChange={(e) => {
                    const next = [...drafts];
                    next[idx] = { ...next[idx], translated_text: e.target.value };
                    setDrafts(next);
                  }}
                />
              </div>
            );
          })
        )}
      </div>

      {drafts.length > 0 && (
        <div className="inspector-footer">
          <button
            type="button"
            className="btn btn-primary"
            style={{ flex: 1 }}
            onClick={handleApply}
            disabled={applying}
          >
            {applying ? (
              <>
                <LoaderIcon size={15} /> Applying changes…
              </>
            ) : (
              "Apply changes"
            )}
          </button>
        </div>
      )}
    </aside>
  );
};
