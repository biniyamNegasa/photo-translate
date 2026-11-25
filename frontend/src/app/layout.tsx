import type { LayoutProps, Metadata } from "@farm.js/core";
import "@fontsource-variable/geist";
import "./globals.css";

export const metadata: Metadata = {
  title: "Photo Translate",
  description:
    "Translate text inside memes, drawings, comics, and photos locally without altering the artwork.",
  icons: {
    icon: [{ url: "/favicon.svg", type: "image/svg+xml", sizes: "any" }],
  },
};

export default function RootLayout({ children }: LayoutProps) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="color-scheme" content="light dark" />
        <meta name="referrer" content="no-referrer" />
      </head>
      <body>{children}</body>
    </html>
  );
}
