import { cn } from "@/lib/utils";

export function NumberPill({
  label,
  value,
  mono = false,
  className,
}: {
  label: string;
  value: string;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/20 px-3 py-2 shadow-[0_10px_30px_-18px_rgba(0,0,0,0.85)] backdrop-blur",
        className,
      )}
    >
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={cn("text-xs font-semibold text-foreground", mono ? "font-mono" : "")}>
        {value}
      </div>
    </div>
  );
}
