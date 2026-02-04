import { Link } from "wouter";
import { GlowButton } from "@/components/GlowButton";
import { BrandMark } from "@/components/BrandMark";
import { ArrowLeft } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen mesh-bg">
      <div className="mx-auto flex min-h-screen max-w-3xl flex-col justify-center px-4 sm:px-6 lg:px-8">
        <div className="rounded-3xl border border-white/10 bg-black/30 p-6 sm:p-10 shadow-[0_40px_120px_-70px_rgba(0,0,0,0.95)] backdrop-blur noise-overlay">
          <BrandMark />
          <div className="mt-8">
            <h1 className="font-display text-4xl text-foreground">404</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              This route doesn’t exist. Head back to the studio.
            </p>
          </div>

          <div className="mt-8">
            <Link href="/" className="inline-block">
              <GlowButton variant="primary">
                <ArrowLeft className="size-4" />
                Back to Planet Studio
              </GlowButton>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
