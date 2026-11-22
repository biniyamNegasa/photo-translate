"use client";

import React from "react";
import type { LanguageDTO, TranslatorInfo } from "../lib/types";

interface LanguageBarProps {
  languages: LanguageDTO[];
  translators: TranslatorInfo[];
  sourceLang: string;
  targetLang: string;
  translator: string;
  onSourceLangChange: (val: string) => void;
  onTargetLangChange: (val: string) => void;
  onTranslatorChange: (val: string) => void;
  disabled?: boolean;
}

export const LanguageBar: React.FC<LanguageBarProps> = ({
  languages,
  translators,
  sourceLang,
  targetLang,
  translator,
  onSourceLangChange,
  onTargetLangChange,
  onTranslatorChange,
  disabled,
}) => {
  return (
    <div className="settings-strip" aria-label="Translation settings">
      <div className="inline-field">
        <label htmlFor="source-lang">From</label>
        <select
          id="source-lang"
          className="select-input"
          value={sourceLang}
          onChange={(e) => onSourceLangChange(e.target.value)}
          disabled={disabled}
        >
          <option value="auto">Auto-detect</option>
          {languages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name} ({lang.native_name})
            </option>
          ))}
        </select>
      </div>

      <div className="inline-field">
        <label htmlFor="target-lang">To</label>
        <select
          id="target-lang"
          className="select-input"
          value={targetLang}
          onChange={(e) => onTargetLangChange(e.target.value)}
          disabled={disabled}
        >
          {languages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name} ({lang.native_name})
            </option>
          ))}
        </select>
      </div>

      <div className="inline-field">
        <label htmlFor="translator-engine">Translator</label>
        <select
          id="translator-engine"
          className="select-input"
          value={translator}
          onChange={(e) => onTranslatorChange(e.target.value)}
          disabled={disabled}
        >
          {translators.map((t) => (
            <option key={t.id} value={t.id} disabled={!t.available}>
              {t.name}
              {!t.available ? " (not configured)" : ""}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};
