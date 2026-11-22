export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export type ItemStatus =
  | 'queued'
  | 'detecting'
  | 'inpainting'
  | 'translating'
  | 'rendering'
  | 'completed'
  | 'no_text'
  | 'failed';

export interface TextStyle {
  fg_color: [number, number, number];
  bg_color: [number, number, number];
  has_stroke: boolean;
  stroke_color: [number, number, number];
  stroke_width: number;
  is_bold: boolean;
  is_uppercase: boolean;
  angle: number;
  alignment: string;
  uniform_bg: boolean;
}

export interface TextBlock {
  id: string;
  polygon: number[][];
  bbox: [number, number, number, number];
  expanded_bbox: [number, number, number, number];
  source_text: string;
  translated_text: string;
  source_lang: string;
  confidence: number;
  enabled: boolean;
  line_count: number;
  style: TextStyle;
}

export interface JobItemInfo {
  item_id: string;
  filename: string;
  width: number;
  height: number;
  format: string;
  status: ItemStatus;
  stage_message: string;
  error?: string | null;
  blocks: TextBlock[];
  original_url: string;
  translated_url: string;
}

export interface JobInfo {
  job_id: string;
  status: JobStatus;
  source_lang: string;
  target_lang: string;
  translator: string;
  created_at: number;
  expires_at: number;
  total_items: number;
  completed_items: number;
  items: JobItemInfo[];
}

export interface LanguageDTO {
  code: string;
  name: string;
  native_name: string;
  script: string;
  rtl: boolean;
}

export interface TranslatorInfo {
  id: string;
  name: string;
  description: string;
  available: boolean;
  is_local: boolean;
}

export interface SystemStatus {
  lama_downloaded: boolean;
  raqm_available: boolean;
  device: string;
  translators: TranslatorInfo[];
}

export interface BlockUpdate {
  id: string;
  translated_text?: string;
  enabled?: boolean;
}
