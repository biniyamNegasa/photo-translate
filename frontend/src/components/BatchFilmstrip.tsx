"use client";

import React from "react";
import type { JobItemInfo } from "../lib/types";
import { AlertCircleIcon, CheckCircleIcon, LoaderIcon, MinusCircleIcon } from "./Icons";

interface BatchFilmstripProps {
  items: JobItemInfo[];
  selectedItemId: string;
  onSelect: (itemId: string) => void;
}

export const BatchFilmstrip: React.FC<BatchFilmstripProps> = ({
  items,
  selectedItemId,
  onSelect,
}) => {
  if (items.length <= 1) return null;

  const renderBadge = (item: JobItemInfo) => {
    if (item.status === "completed") {
      return <CheckCircleIcon size={13} color="var(--success)" />;
    }
    if (item.status === "no_text") {
      return <MinusCircleIcon size={13} color="var(--ink-muted)" />;
    }
    if (item.status === "failed") {
      return <AlertCircleIcon size={13} color="var(--danger)" />;
    }
    return <LoaderIcon size={13} color="var(--accent)" />;
  };

  return (
    <div className="filmstrip" role="region" aria-label="Uploaded pages">
      {items.map((item) => {
        const isCurrent = item.item_id === selectedItemId;
        return (
          <button
            key={item.item_id}
            type="button"
            className={`thumb-card ${isCurrent ? "active" : ""}`}
            aria-current={isCurrent ? "true" : undefined}
            onClick={() => onSelect(item.item_id)}
            title={`${item.filename}: ${item.stage_message}`}
          >
            <div className="thumb-img-wrap">
              <img
                src={item.status === "completed" ? item.translated_url : item.original_url}
                alt={item.filename}
                loading="lazy"
              />
              <span className="thumb-status" aria-hidden="true">
                {renderBadge(item)}
              </span>
            </div>
            <div className="thumb-name">{item.filename}</div>
          </button>
        );
      })}
    </div>
  );
};
