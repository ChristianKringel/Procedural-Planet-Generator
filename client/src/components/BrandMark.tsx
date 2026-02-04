import { Globe } from "lucide-react";

export function BrandMark() {
  return (
    <div className="flex items-center gap-3">
      <div
        className="
          relative grid size-10 place-items-center rounded-2xl
          bg-gradient-to-br from-primary/22 via-primary/10 to-accent/10
          border border-white/10 shadow-[0_18px_50px_-22px_rgba(0,0,0,0.7)]
        "
        aria-hidden="true"
      >
        <div className="absolute inset-0 rounded-2xl noise-overlay" />
        <Globe className="size-5 text-primary drop-shadow" />
      </div>

      <div className="leading-tight">
        <div className="font-display text-lg tracking-tight text-foreground">
          Procedural Planet
        </div>
        <div className="text-xs text-muted-foreground">WebGL2 • interactive sandbox</div>
      </div>
    </div>
  );
}
