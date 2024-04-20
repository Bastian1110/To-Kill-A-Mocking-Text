import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kill A Mocking Text",
  description: "Text smilarity detector",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
