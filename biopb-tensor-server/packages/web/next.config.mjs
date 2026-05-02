/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  reactStrictMode: true,
  transpilePackages: ["@biopb/tensor-flight-client"],
};

export default nextConfig;
