"use client";

import React from "react";

interface IconProps {
  size?: number;
  className?: string;
  color?: string;
}

const baseProps = (size: number, className?: string, color?: string) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: color || "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  className,
  "aria-hidden": true,
});

export const LanguagesIcon: React.FC<IconProps> = ({ size = 20, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="m5 8 6 6" />
    <path d="m4 14 6-6 2-3" />
    <path d="M2 5h12" />
    <path d="M7 2h1" />
    <path d="m22 22-5-10-5 10" />
    <path d="M14 18h6" />
  </svg>
);

export const UploadIcon: React.FC<IconProps> = ({ size = 24, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" x2="12" y1="3" y2="15" />
  </svg>
);

export const ImagePlusIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7" />
    <line x1="16" x2="22" y1="5" y2="5" />
    <line x1="19" x2="19" y1="2" y2="8" />
    <circle cx="9" cy="9" r="2" />
    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
  </svg>
);

export const ArchiveIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <rect width="20" height="5" x="2" y="3" rx="1" />
    <path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8" />
    <path d="M10 12h4" />
  </svg>
);

export const XIcon: React.FC<IconProps> = ({ size = 16, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M18 6 6 18" />
    <path d="m6 6 12 12" />
  </svg>
);

export const LoaderIcon: React.FC<IconProps> = ({ size = 18, className, color }) => (
  <svg {...baseProps(size, className ? `spin ${className}` : "spin", color)}>
    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
  </svg>
);

export const SparklesIcon: React.FC<IconProps> = ({ size = 16, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" />
  </svg>
);

export const GripVerticalIcon: React.FC<IconProps> = ({ size = 16, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <circle cx="9" cy="12" r="1" />
    <circle cx="9" cy="5" r="1" />
    <circle cx="9" cy="19" r="1" />
    <circle cx="15" cy="12" r="1" />
    <circle cx="15" cy="5" r="1" />
    <circle cx="15" cy="19" r="1" />
  </svg>
);

export const CheckCircleIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <circle cx="12" cy="12" r="10" />
    <path d="m9 12 2 2 4-4" />
  </svg>
);

export const MinusCircleIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <circle cx="12" cy="12" r="10" />
    <path d="M8 12h8" />
  </svg>
);

export const AlertCircleIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <circle cx="12" cy="12" r="10" />
    <line x1="12" x2="12" y1="8" y2="12" />
    <line x1="12" x2="12.01" y1="16" y2="16" />
  </svg>
);

export const DownloadIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" x2="12" y1="15" y2="3" />
  </svg>
);

export const PlusIcon: React.FC<IconProps> = ({ size = 15, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M5 12h14" />
    <path d="M12 5v14" />
  </svg>
);

export const SlidersIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <line x1="21" x2="14" y1="4" y2="4" />
    <line x1="10" x2="3" y1="4" y2="4" />
    <line x1="21" x2="12" y1="12" y2="12" />
    <line x1="8" x2="3" y1="12" y2="12" />
    <line x1="21" x2="16" y1="20" y2="20" />
    <line x1="12" x2="3" y1="20" y2="20" />
    <line x1="14" x2="14" y1="2" y2="6" />
    <line x1="8" x2="8" y1="10" y2="14" />
    <line x1="16" x2="16" y1="18" y2="22" />
  </svg>
);

export const RefreshIcon: React.FC<IconProps> = ({ size = 15, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
    <path d="M8 16H3v5" />
  </svg>
);

export const ZoomInIcon: React.FC<IconProps> = ({ size = 15, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <circle cx="11" cy="11" r="8" />
    <line x1="21" x2="16.65" y1="21" y2="16.65" />
    <line x1="11" x2="11" y1="8" y2="14" />
    <line x1="8" x2="14" y1="11" y2="11" />
  </svg>
);

export const ZoomOutIcon: React.FC<IconProps> = ({ size = 15, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <circle cx="11" cy="11" r="8" />
    <line x1="21" x2="16.65" y1="21" y2="16.65" />
    <line x1="8" x2="14" y1="11" y2="11" />
  </svg>
);

export const MaximizeIcon: React.FC<IconProps> = ({ size = 15, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <polyline points="15 3 21 3 21 9" />
    <polyline points="9 21 3 21 3 15" />
    <line x1="21" x2="14" y1="3" y2="10" />
    <line x1="3" x2="10" y1="21" y2="14" />
  </svg>
);

export const MinimizeIcon: React.FC<IconProps> = ({ size = 15, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <polyline points="4 14 10 14 10 20" />
    <polyline points="20 10 14 10 14 4" />
    <line x1="14" x2="21" y1="10" y2="3" />
    <line x1="3" x2="10" y1="21" y2="14" />
  </svg>
);

export const BellIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
  </svg>
);

export const BellOffIcon: React.FC<IconProps> = ({ size = 14, className, color }) => (
  <svg {...baseProps(size, className, color)}>
    <path d="M8.7 3A6 6 0 0 1 18 8a21.3 21.3 0 0 0 .6 5" />
    <path d="M17 17H3s3-2 3-9a4.67 4.67 0 0 1 .3-1.7" />
    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    <line x1="2" x2="22" y1="2" y2="22" />
  </svg>
);
