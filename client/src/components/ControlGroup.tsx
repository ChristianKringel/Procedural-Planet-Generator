import * as React from "react";
import { cn } from "@/lib/utils";

export function ControlGroup({
  title,
  description,
  children,
  right,
  className,
}: {
  title: string;
  description?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        "rounded-2xl border border-card-border/80 bg-card/60 p-4 shadow-[0_18px_60px_-40px_rgba(0,0,0,0.75)] backdrop-blur-sm noise-overlay",
        className,
      )}
    >
      <header className="flex items-start justify-between gap-4">
        <div>
          <h3 className="font-display text-base text-foreground">{title}</h3>
          {description ? (
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{description}</p>
          ) : null}
        </div>
        {right ? <div className="shrink-0">{right}</div> : null}
      </header>

      <div className="mt-4">{children}</div>
    </section>
  );
}
