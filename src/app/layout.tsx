import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AME · Local Mesh Capture",
  description: "A browser-local research preview for descriptive facial landmark capture.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
