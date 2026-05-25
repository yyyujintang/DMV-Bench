/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    // Symlinked /public/images can resolve outside the standard public root on some FS layouts.
    // Use unoptimized for now to skip the optimizer; we can flip back on Vercel.
    unoptimized: true,
  },
};
export default nextConfig;
