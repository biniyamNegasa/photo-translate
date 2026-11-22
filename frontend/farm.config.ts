import { defineConfig } from "@farm.js/core";

export default defineConfig({
  devtools: false,
  theme: {
    default: "system",
  },
  images: {
    provider: "none",
  },
  docs: {
    enabled: false,
  },
});
