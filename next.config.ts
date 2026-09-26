import type { NextConfig } from "next";

// This is a parallel static build, not yet the GitHub Pages deployment.
const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
};

export default nextConfig;
