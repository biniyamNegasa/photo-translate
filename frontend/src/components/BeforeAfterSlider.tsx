"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import type { JobItemInfo } from "../lib/types";
import {
  GripVerticalIcon,
  LoaderIcon,
  MaximizeIcon,
  MinimizeIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from "./Icons";

export type ViewMode = "split" | "translated" | "original";

interface BeforeAfterSliderProps {
  item: JobItemInfo;
  viewMode: ViewMode;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export const BeforeAfterSlider: React.FC<BeforeAfterSliderProps> = ({ item, viewMode }) => {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const [splitPct, setSplitPct] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const dragModeRef = useRef<"none" | "split" | "pan">("none");
  const panStartRef = useRef({ clientX: 0, clientY: 0, panX: 0, panY: 0 });

  const isWorking =
    item.status === "queued" ||
    item.status === "detecting" ||
    item.status === "inpainting" ||
    item.status === "translating" ||
    item.status === "rendering";

  const translatedReady = item.status === "completed" || item.status === "no_text";

  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (expanded && !dlg.open) {
      dlg.showModal();
    } else if (!expanded && dlg.open) {
      dlg.close();
    }
  }, [expanded]);

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [item.item_id]);

  const clampPan = useCallback((nextX: number, nextY: number, scale: number) => {
    if (scale <= 1) return { x: 0, y: 0 };
    const stage = stageRef.current;
    const frame = frameRef.current;
    if (!stage || !frame) return { x: nextX, y: nextY };
    const maxX = Math.max(0, (frame.offsetWidth * scale - stage.clientWidth) / 2 + 64);
    const maxY = Math.max(0, (frame.offsetHeight * scale - stage.clientHeight) / 2 + 64);
    return {
      x: clamp(nextX, -maxX, maxX),
      y: clamp(nextY, -maxY, maxY),
    };
  }, []);

  const applyZoom = useCallback(
    (targetZoom: number, anchorClientX?: number, anchorClientY?: number) => {
      const nextZoom = clamp(Number(targetZoom.toFixed(3)), MIN_ZOOM, MAX_ZOOM);
      setZoom((prevZoom) => {
        if (nextZoom <= 1) {
          setPan({ x: 0, y: 0 });
          return 1;
        }
        const stage = stageRef.current;
        if (stage && anchorClientX !== undefined && anchorClientY !== undefined) {
          const rect = stage.getBoundingClientRect();
          const cx = anchorClientX - (rect.left + rect.width / 2);
          const cy = anchorClientY - (rect.top + rect.height / 2);
          const ratio = nextZoom / prevZoom;
          setPan((prevPan) =>
            clampPan(
              cx - (cx - prevPan.x) * ratio,
              cy - (cy - prevPan.y) * ratio,
              nextZoom
            )
          );
        } else {
          const ratio = nextZoom / prevZoom;
          setPan((prevPan) => clampPan(prevPan.x * ratio, prevPan.y * ratio, nextZoom));
        }
        return nextZoom;
      });
    },
    [clampPan]
  );

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const zoomToActualPixels = useCallback(() => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !img.clientWidth) return;
    const exactScale = img.naturalWidth / img.clientWidth;
    applyZoom(clamp(exactScale, 1, MAX_ZOOM));
  }, [applyZoom]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const onWheel = (e: WheelEvent) => {
      if (isWorking) return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
      setZoom((current) => {
        const nextZoom = clamp(Number((current * factor).toFixed(3)), MIN_ZOOM, MAX_ZOOM);
        if (nextZoom <= 1) {
          setPan({ x: 0, y: 0 });
          return 1;
        }
        const rect = stage.getBoundingClientRect();
        const cx = e.clientX - (rect.left + rect.width / 2);
        const cy = e.clientY - (rect.top + rect.height / 2);
        const ratio = nextZoom / current;
        setPan((prevPan) =>
          clampPan(cx - (cx - prevPan.x) * ratio, cy - (cy - prevPan.y) * ratio, nextZoom)
        );
        return nextZoom;
      });
    };
    stage.addEventListener("wheel", onWheel, { passive: false });
    return () => stage.removeEventListener("wheel", onWheel);
  }, [isWorking, expanded, clampPan]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "TEXTAREA" || target.tagName === "INPUT" || target.tagName === "SELECT")
      ) {
        return;
      }
      if (e.key === "=" || e.key === "+") {
        e.preventDefault();
        applyZoom(zoom * 1.25);
      } else if (e.key === "-") {
        e.preventDefault();
        applyZoom(zoom / 1.25);
      } else if (e.key === "0") {
        e.preventDefault();
        resetView();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [zoom, applyZoom, resetView]);

  const updateSplitFromClientX = (clientX: number) => {
    const el = frameRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / Math.max(1, rect.width)) * 100;
    setSplitPct(clamp(pct, 0, 100));
  };

  const handleDividerPointerDown = (e: React.PointerEvent) => {
    if (viewMode !== "split" || isWorking || !translatedReady) return;
    e.stopPropagation();
    dragModeRef.current = "split";
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    updateSplitFromClientX(e.clientX);
  };

  const handleStagePointerDown = (e: React.PointerEvent) => {
    if (isWorking) return;
    if ((e.target as HTMLElement).closest(".zoom-controls")) return;

    if (zoom > 1) {
      dragModeRef.current = "pan";
      setIsPanning(true);
      panStartRef.current = {
        clientX: e.clientX,
        clientY: e.clientY,
        panX: pan.x,
        panY: pan.y,
      };
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } else if (viewMode === "split" && translatedReady) {
      dragModeRef.current = "split";
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      updateSplitFromClientX(e.clientX);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (dragModeRef.current === "split") {
      updateSplitFromClientX(e.clientX);
    } else if (dragModeRef.current === "pan") {
      const dx = e.clientX - panStartRef.current.clientX;
      const dy = e.clientY - panStartRef.current.clientY;
      setPan(clampPan(panStartRef.current.panX + dx, panStartRef.current.panY + dy, zoom));
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (dragModeRef.current === "none") return;
    dragModeRef.current = "none";
    setIsPanning(false);
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    if (isWorking) return;
    if ((e.target as HTMLElement).closest(".zoom-controls, .compare-divider")) return;
    if (zoom > 1.05) {
      resetView();
    } else {
      applyZoom(2.5, e.clientX, e.clientY);
    }
  };

  const handleSliderKeyDown = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 1 : 5;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      setSplitPct((prev) => clamp(prev - step, 0, 100));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      setSplitPct((prev) => clamp(prev + step, 0, 100));
    } else if (e.key === "Home") {
      e.preventDefault();
      setSplitPct(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setSplitPct(100);
    }
  };

  const stageCursor =
    zoom > 1
      ? isPanning
        ? "grabbing"
        : "grab"
      : viewMode === "split" && translatedReady
        ? "ew-resize"
        : "zoom-in";

  const touchAction = zoom > 1 ? "none" : "pan-y";

  const renderStage = (inModal: boolean) => (
    <div
      ref={inModal === expanded ? stageRef : undefined}
      className={`stage-container ${inModal ? "expanded" : ""}`}
      style={{ cursor: stageCursor, touchAction }}
      onPointerDown={handleStagePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onDoubleClick={handleDoubleClick}
    >
      {viewMode === "split" && translatedReady && (
        <>
          <span className="compare-label left">Original</span>
          <span className="compare-label right">Translated</span>
        </>
      )}
      {viewMode === "original" && <span className="compare-label left">Original</span>}
      {viewMode === "translated" && translatedReady && (
        <span className="compare-label right">
          {item.status === "no_text" ? "No text found" : "Translated"}
        </span>
      )}

      <div
        ref={inModal === expanded ? frameRef : undefined}
        className={`compare-frame ${isPanning || dragModeRef.current === "split" ? "is-dragging" : ""}`}
        style={{
          transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
          transformOrigin: "center center",
          willChange: isPanning || zoom > 1 ? "transform" : "auto",
        }}
      >
        <img
          ref={inModal === expanded ? imgRef : undefined}
          src={
            viewMode === "original" || !translatedReady
              ? item.original_url
              : item.translated_url
          }
          alt={
            viewMode === "original"
              ? `Original ${item.filename}`
              : `Translated ${item.filename}`
          }
          className={`compare-img ${inModal ? "expanded" : ""}`}
          draggable={false}
        />

        {viewMode === "split" && translatedReady && (
          <>
            <div
              className="compare-overlay"
              style={{ clipPath: `inset(0 ${100 - splitPct}% 0 0)` }}
            >
              <img src={item.original_url} alt="" draggable={false} />
            </div>

            <div
              className="compare-divider"
              style={{
                left: `${splitPct}%`,
                width: `${Math.max(1.5, 2 / zoom)}px`,
              }}
              onPointerDown={handleDividerPointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            >
              <div
                role="slider"
                tabIndex={0}
                aria-label="Comparison split position"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(splitPct)}
                aria-valuetext={`${Math.round(splitPct)}% original, ${Math.round(100 - splitPct)}% translated`}
                className="compare-handle"
                style={{
                  transform: `translate(-50%, -50%) scale(${1 / zoom})`,
                }}
                onKeyDown={handleSliderKeyDown}
                title="Drag or use arrow keys to compare original and translated image"
              >
                <GripVerticalIcon size={15} />
              </div>
            </div>
          </>
        )}

        {isWorking && (
          <div className="stage-loading-overlay">
            <LoaderIcon size={26} />
            <span>{item.stage_message || "Working…"}</span>
          </div>
        )}
      </div>

      <div className="zoom-controls" role="group" aria-label="Zoom controls">
        <button
          type="button"
          className="zoom-btn"
          onClick={() => applyZoom(zoom / 1.3)}
          disabled={zoom <= MIN_ZOOM}
          title="Zoom out (-)"
          aria-label="Zoom out"
        >
          <ZoomOutIcon size={15} />
        </button>

        <button
          type="button"
          className="zoom-readout"
          onClick={resetView}
          title="Reset to fit (0)"
        >
          {Math.round(zoom * 100)}%
        </button>

        <button
          type="button"
          className="zoom-btn"
          onClick={() => applyZoom(zoom * 1.3)}
          disabled={zoom >= MAX_ZOOM}
          title="Zoom in (+)"
          aria-label="Zoom in"
        >
          <ZoomInIcon size={15} />
        </button>

        <span className="zoom-sep" aria-hidden="true" />

        <button
          type="button"
          className="zoom-text-btn"
          onClick={resetView}
          disabled={zoom === 1 && pan.x === 0 && pan.y === 0}
          title="Fit image to canvas"
        >
          Fit
        </button>

        <button
          type="button"
          className="zoom-text-btn"
          onClick={zoomToActualPixels}
          title={`View at actual size (${item.width}×${item.height})`}
        >
          1:1
        </button>

        <span className="zoom-sep" aria-hidden="true" />

        <button
          type="button"
          className="zoom-btn"
          onClick={() => {
            setExpanded((v) => !v);
            resetView();
          }}
          title={inModal ? "Exit fullscreen (Esc)" : "Fullscreen"}
          aria-label={inModal ? "Exit fullscreen" : "Open fullscreen"}
        >
          {inModal ? <MinimizeIcon size={15} /> : <MaximizeIcon size={15} />}
        </button>
      </div>
    </div>
  );

  return (
    <>
      {!expanded && renderStage(false)}
      <dialog
        ref={dialogRef}
        className="viewer-fullscreen-dialog"
        aria-label={`Fullscreen view of ${item.filename}`}
        onClose={() => {
          setExpanded(false);
          resetView();
        }}
      >
        {expanded && renderStage(true)}
      </dialog>
    </>
  );
};
