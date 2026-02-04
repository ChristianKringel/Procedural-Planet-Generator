import * as React from "react";
import { cn } from "@/lib/utils";

export function GlowButton({
  variant = "primary",
  size = "md",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
}) {
  const base =
    "relative inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition-all duration-200 ease-out focus-ring disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none";

  const sizes = {
    sm: "px-3.5 py-2 text-sm",
    md: "px-4.5 py-2.5 text-sm",
  }[size];

  const variants = {
    primary:
      "text-primary-foreground bg-gradient-to-b from-primary to-primary/80 shadow-[0_18px_55px_-24px_hsl(var(--primary)/0.65)] border border-white/10 hover:-translate-y-0.5 hover:shadow-[0_26px_75px_-30px_hsl(var(--primary)/0.75)] active:translate-y-0 active:shadow-[0_18px_55px_-28px_hsl(var(--primary)/0.55)]",
    secondary:
      "text-foreground bg-gradient-to-b from-secondary/70 to-secondary/40 border border-white/10 shadow-[0_18px_60px_-40px_rgba(0,0,0,0.85)] hover:-translate-y-0.5 hover:border-white/15 hover:bg-secondary/60 active:translate-y-0",
    ghost:
      "text-foreground/90 bg-transparent border border-white/10 hover:bg-white/5 hover:border-white/15 active:bg-white/7",
    danger:
      "text-destructive-foreground bg-gradient-to-b from-destructive to-destructive/85 border border-white/10 shadow-[0_18px_55px_-24px_rgba(239,68,68,0.35)] hover:-translate-y-0.5 hover:shadow-[0_26px_75px_-30px_rgba(239,68,68,0.45)] active:translate-y-0",
  }[variant];

  return (
    <button
      className={cn(base, sizes, variants, "shine", className)}
      {...props}
    >
      {props.children}
    </button>
  );
}
