import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable strict mode for better development warnings
  reactStrictMode: true,

  // Configure external packages if needed
  transpilePackages: [],

  // Environment variables exposed to the browser
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

export default nextConfig;
