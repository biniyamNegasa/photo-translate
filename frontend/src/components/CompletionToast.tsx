"use client";

import React, { useEffect, useId, useRef } from "react";
import { AlertCircleIcon, CheckCircleIcon, XIcon } from "./Icons";

export interface ToastNotice {
  id: string;
  headline: string;
  detail: string;
  isFailure: boolean;
}

interface CompletionToastProps {
  notice: ToastNotice | null;
  onDismiss: () => void;
}

function supportsNativePopover(): boolean {
  return (
    typeof HTMLElement !== "undefined" &&
    Object.prototype.hasOwnProperty.call(HTMLElement.prototype, "popover")
  );
}

export const CompletionToast: React.FC<CompletionToastProps> = ({ notice, onDismiss }) => {
  const toastId = useId();
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const priorFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const el = popoverRef.current;
    if (!el) return;

    const showToast = () => {
      // Do not surface while a modal <dialog> (e.g. fullscreen viewer) is open
      if (document.querySelector("dialog[open]")) return;
      if (supportsNativePopover()) {
        try {
          if (!el.matches(":popover-open")) {
            el.showPopover();
          }
        } catch {
          el.classList.add("toast-open");
        }
      } else {
        el.classList.add("toast-open");
      }
    };

    const hideToast = () => {
      if (supportsNativePopover()) {
        try {
          if (el.matches(":popover-open")) {
            el.hidePopover();
          }
        } catch {
          el.classList.remove("toast-open");
        }
      } else {
        el.classList.remove("toast-open");
      }
    };

    if (!notice) {
      hideToast();
      return;
    }

    priorFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;

    if (document.querySelector("dialog[open]")) {
      // Wait for open modal dialog to close before surfacing the toast
      const observer = new MutationObserver(() => {
        if (!document.querySelector("dialog[open]")) {
          observer.disconnect();
          showToast();
        }
      });
      observer.observe(document.body, {
        subtree: true,
        attributes: true,
        attributeFilter: ["open"],
      });
      return () => observer.disconnect();
    }

    showToast();
    return undefined;
  }, [notice]);

  const handleClose = () => {
    const el = popoverRef.current;
    const wasFocusedInside = el ? el.contains(document.activeElement) : false;

    if (el && supportsNativePopover()) {
      try {
        if (el.matches(":popover-open")) {
          el.hidePopover();
        }
      } catch {
        el.classList.remove("toast-open");
      }
    } else if (el) {
      el.classList.remove("toast-open");
    }

    onDismiss();

    if (wasFocusedInside && priorFocusRef.current?.isConnected) {
      priorFocusRef.current.focus();
    }
  };

  return (
    <div role="region" aria-label="Notifications" className="toast-region">
      <div
        ref={popoverRef}
        id={toastId}
        {...({ popover: "manual" } as Record<string, string>)}
        className={`completion-toast ${notice?.isFailure ? "is-failure" : "is-success"}`}
      >
        {notice && (
          <>
            <span className="completion-toast-icon" aria-hidden="true">
              {notice.isFailure ? (
                <AlertCircleIcon size={17} />
              ) : (
                <CheckCircleIcon size={17} />
              )}
            </span>
            <div className="completion-toast-copy">
              <strong className="completion-toast-title">{notice.headline}</strong>
              {notice.detail && (
                <span className="completion-toast-detail">{notice.detail}</span>
              )}
            </div>
            <button
              type="button"
              className="btn btn-quiet btn-icon completion-toast-close"
              {...({
                popoverTarget: toastId,
                popoverTargetAction: "hide",
              } as Record<string, string>)}
              onClick={handleClose}
              aria-label="Dismiss notification"
            >
              <XIcon size={15} />
            </button>
          </>
        )}
      </div>
    </div>
  );
};
