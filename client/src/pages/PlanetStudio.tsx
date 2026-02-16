import * as React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Helmet } from "@/components/Helmet";
import { SidebarProvider, Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent, SidebarGroupLabel, SidebarHeader, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/hooks/use-toast";
import { BrandMark } from "@/components/BrandMark";
import { ControlGroup } from "@/components/ControlGroup";
import { GlowButton } from "@/components/GlowButton";
import { NumberPill } from "@/components/NumberPill";
import { PlanetRenderer } from "@/lib/planet/renderer";
import { planetSettingsSchema, type NoiseType, type PlanetSettings } from "@shared/schema";
import { useValidatePlanetSettings } from "@/hooks/use-planet";
import { Menu, RefreshCcw, Rocket, RotateCw, Sparkles, SunMoon, Wand2 } from "lucide-react";
import { cn } from "@/lib/utils";

const DEFAULT_SETTINGS: PlanetSettings = {
  seed: "aurora-001",
  subdivisions: 120,
  waterThreshold: 0.0,
  noiseStrength: 1.5,
  noiseType: "simplex",
  objectCount: 420,
  shadowsEnabled: true,
  missileDuration: 1.2,
};

function useDebouncedValue<T>(value: T, delayMs: number) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(t);
  }, [value, delayMs]);
  return debounced;
}

function clampNumber(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

export default function PlanetStudio() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<PlanetRenderer | null>(null);

  const [settings, setSettings] = useState<PlanetSettings>(DEFAULT_SETTINGS);
  const [autoRotate, setAutoRotate] = useState(true);

  const debouncedLive = useDebouncedValue(settings, 180);

  const validateMutation = useValidatePlanetSettings();

  const [hud, setHud] = useState({ fps: 60, triPlanet: 0, triObjects: 0, seed: settings.seed });

  // Live update: rebuild only when subdivisions/seed/noiseType change; else update uniforms.
  const rebuildKey = useMemo(() => {
    return `${debouncedLive.seed}|${debouncedLive.subdivisions}|${debouncedLive.noiseType}`;
  }, [debouncedLive.seed, debouncedLive.subdivisions, debouncedLive.noiseType]);

  const redistributeKey = useMemo(() => {
    return `${debouncedLive.objectCount}|${debouncedLive.seed}|${debouncedLive.noiseType}|${debouncedLive.noiseStrength}|${debouncedLive.waterThreshold}`;
  }, [
    debouncedLive.objectCount,
    debouncedLive.seed,
    debouncedLive.noiseType,
    debouncedLive.noiseStrength,
    debouncedLive.waterThreshold,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      rendererRef.current = new PlanetRenderer(canvas, settings);
      // Sync autoRotate state with renderer on initialization
      rendererRef.current.setAutoRotate(autoRotate);
    } catch (e) {
      console.error(e);
      toast({
        title: "WebGL2 initialization failed",
        description:
          e instanceof Error ? e.message : "Your browser/GPU might not support WebGL2.",
        variant: "destructive",
      });
      return;
    }

    let raf = 0;
    const tick = () => {
      const r = rendererRef.current;
      if (r) setHud(r.getStats());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    const onResize = () => rendererRef.current?.resize();
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Apply debounced settings
  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;

    const parsed = planetSettingsSchema.safeParse(debouncedLive);
    if (!parsed.success) {
      console.warn("[Zod] local settings invalid:", parsed.error.format());
      return;
    }

    r.setSettings(parsed.data, {
      rebuild: true, // will be gated by rebuildKey effect below
      redistribute: true,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rebuildKey, redistributeKey]);

  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;

    // Always update toggles and continuous sliders (no rebuild)
    const parsed = planetSettingsSchema.safeParse(debouncedLive);
    if (!parsed.success) return;

    r.setSettings(parsed.data, {
      rebuild: false,
      redistribute: false,
    });
  }, [debouncedLive.shadowsEnabled, debouncedLive.noiseStrength, debouncedLive.waterThreshold, debouncedLive.missileDuration]);

  const onRegenerate = async () => {
    const local = planetSettingsSchema.safeParse(settings);
    if (!local.success) {
      toast({
        title: "Invalid settings",
        description: "Please review your inputs.",
        variant: "destructive",
      });
      return;
    }

    // Try server validation; fallback to local on failure.
    try {
      const validated = await validateMutation.mutateAsync(local.data);
      setSettings(validated);

      rendererRef.current?.setSettings(validated, { rebuild: true, redistribute: true });

      toast({
        title: "Regenerated",
        description: "Planet mesh and objects were rebuilt from your settings.",
      });
    } catch (e) {
      rendererRef.current?.setSettings(local.data, { rebuild: true, redistribute: true });
      toast({
        title: "Regenerated (local)",
        description:
          e instanceof Error
            ? `Server validation skipped: ${e.message}`
            : "Server validation skipped.",
      });
    }
  };

  const onToggleShadows = () => {
    setSettings((s) => ({ ...s, shadowsEnabled: !s.shadowsEnabled }));
  };

  const onToggleRotation = () => {
    const newValue = !autoRotate;
    setAutoRotate(newValue);
    if (rendererRef.current) {
      rendererRef.current.setAutoRotate(newValue);
    }
  };

  const onCanvasPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 2) return; // only right-click places objects
    e.preventDefault();
    const r = rendererRef.current;
    if (!r) return;

    const pick = r.pick(e.clientX, e.clientY);
    if (!pick.hit) return;

    r.placeObjectFromPick(pick);

    toast({
      title: pick.isWater ? "Boat placed" : "Tree planted",
      description: pick.isWater
        ? "A small boat is now bobbing on the water."
        : "A tree anchored itself on the terrain.",
    });
  };

  return (
    <>
      <Helmet
        title="Procedural Planet Studio"
        description="A premium WebGL2 playground: procedural sphere terrain, shadows, and object placement with ray picking."
      />
      <div className="min-h-screen mesh-bg">
        <SidebarProvider defaultOpen>
          <div className="flex min-h-screen w-full">
            <Sidebar className="border-r border-sidebar-border/80 bg-sidebar/75 backdrop-blur-xl">
              <SidebarHeader className="p-5">
                <div className="animate-float-in">
                  <BrandMark />
                </div>

                <div className="mt-4 rounded-2xl border border-white/10 bg-black/25 p-3 noise-overlay">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Sparkles className="size-4 text-primary" />
                      <div className="text-sm font-semibold text-foreground">Live controls</div>
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      debounced
                    </div>
                  </div>
                  <div className="mt-2 text-xs leading-relaxed text-muted-foreground">
                    Changes apply smoothly; use <span className="text-foreground">Regenerate</span> to
                    fully rebuild & validate.
                  </div>
                </div>
              </SidebarHeader>

              <SidebarContent className="px-5 pb-5">
                <SidebarGroup>
                  <SidebarGroupLabel className="text-xs tracking-wide text-muted-foreground">
                    Planet
                  </SidebarGroupLabel>
                  <SidebarGroupContent className="space-y-4">
                    <ControlGroup
                      title="Seed"
                      description="Determines terrain and object distribution. Any string works."
                      right={
                        <div className="rounded-full border border-white/10 bg-black/20 px-2 py-1 text-[11px] text-muted-foreground">
                          deterministic
                        </div>
                      }
                    >
                      <Label className="sr-only" htmlFor="seed">
                        Seed
                      </Label>
                      <Input
                        id="seed"
                        data-testid="seed-input"
                        value={settings.seed}
                        onChange={(e) => setSettings((s) => ({ ...s, seed: e.target.value }))}
                        className="
                          h-11 rounded-xl bg-background/40 border-2 border-white/10
                          text-foreground placeholder:text-muted-foreground
                          focus:border-primary/60 focus:ring-4 focus:ring-primary/15
                          transition-all
                        "
                        placeholder="e.g. aurora-001"
                      />
                    </ControlGroup>

                    <ControlGroup
                      title="Resolution"
                      description="Subdivisions of the sphere mesh. Higher = smoother, heavier."
                      right={
                        <div className="text-xs font-semibold text-foreground">
                          {settings.subdivisions}
                        </div>
                      }
                    >
                      <Slider
                        data-testid="subdivisions-slider"
                        value={[settings.subdivisions]}
                        min={20}
                        max={200}
                        step={1}
                        onValueChange={(v) =>
                          setSettings((s) => ({ ...s, subdivisions: v[0] ?? s.subdivisions }))
                        }
                      />
                      <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                        <span>20</span>
                        <span>200</span>
                      </div>
                    </ControlGroup>

                    <ControlGroup
                      title="Noise"
                      description="Pick a noise style and intensity to shape continents."
                      right={
                        <div className="text-xs font-semibold text-foreground">
                          {settings.noiseStrength.toFixed(2)}×
                        </div>
                      }
                    >
                      <div className="grid grid-cols-1 gap-3">
                        <div className="grid gap-2">
                          <Label className="text-xs text-muted-foreground">Noise type</Label>
                          <Select
                            value={settings.noiseType}
                            onValueChange={(v) => setSettings((s) => ({ ...s, noiseType: v as NoiseType }))}
                          >
                            <SelectTrigger
                              data-testid="noise-type-dropdown"
                              className="
                                h-11 rounded-xl bg-background/40 border-2 border-white/10
                                focus:border-primary/60 focus:ring-4 focus:ring-primary/15
                                transition-all
                              "
                            >
                              <SelectValue placeholder="Choose noise" />
                            </SelectTrigger>
                            <SelectContent className="border border-white/10 bg-popover/95 backdrop-blur-xl">
                              <SelectItem value="perlin">Perlin</SelectItem>
                              <SelectItem value="simplex">Simplex</SelectItem>
                              <SelectItem value="random">Random</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="grid gap-2">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs text-muted-foreground">Strength</Label>
                            <div className="text-xs font-semibold text-foreground">
                              {settings.noiseStrength.toFixed(2)}
                            </div>
                          </div>
                          <Slider
                            data-testid="noise-strength-slider"
                            value={[settings.noiseStrength]}
                            min={0}
                            max={2}
                            step={0.01}
                            onValueChange={(v) =>
                              setSettings((s) => ({ ...s, noiseStrength: v[0] ?? s.noiseStrength }))
                            }
                          />
                        </div>
                      </div>
                    </ControlGroup>

                    <ControlGroup
                      title="Water vs Land"
                      description="Adjust the threshold where the terrain becomes ocean."
                      right={
                        <div className="text-xs font-semibold text-foreground">
                          {settings.waterThreshold.toFixed(2)}
                        </div>
                      }
                    >
                      <Slider
                        data-testid="threshold-slider"
                        value={[settings.waterThreshold]}
                        min={-1}
                        max={1}
                        step={0.01}
                        onValueChange={(v) =>
                          setSettings((s) => ({ ...s, waterThreshold: v[0] ?? s.waterThreshold }))
                        }
                      />
                    </ControlGroup>

                    <ControlGroup
                      title="Objects"
                      description="Sets the auto-distributed count; click to place more."
                      right={
                        <div className="text-xs font-semibold text-foreground">
                          {settings.objectCount}
                        </div>
                      }
                    >
                      <Slider
                        data-testid="object-count-slider"
                        value={[settings.objectCount]}
                        min={0}
                        max={2000}
                        step={1}
                        onValueChange={(v) =>
                          setSettings((s) => ({ ...s, objectCount: v[0] ?? s.objectCount }))
                        }
                      />
                    </ControlGroup>

                    <ControlGroup
                      title="Shadows"
                      description="Directional sun + shadow map (planet receives, objects cast)."
                      right={
                        <div className="flex items-center gap-2">
                          <SunMoon className="size-4 text-primary" />
                          <Switch
                            data-testid="shadows-toggle"
                            checked={settings.shadowsEnabled}
                            onCheckedChange={(checked) =>
                              setSettings((s) => ({ ...s, shadowsEnabled: !!checked }))
                            }
                          />
                        </div>
                      }
                    >
                      <div className="flex gap-2">
                        <GlowButton
                          data-testid="toggle-shadows-button"
                          variant="secondary"
                          className="w-full"
                          onClick={onToggleShadows}
                        >
                          <SunMoon className="size-4" />
                          Toggle shadows
                        </GlowButton>
                      </div>
                    </ControlGroup>

                    <ControlGroup
                      title="Missile Duration"
                      description="Control how fast missiles travel when launched with 'M' key."
                      right={
                        <div className="flex items-center gap-2">
                          <Rocket className="size-4 text-primary" />
                          <NumberPill
                            value={settings.missileDuration.toFixed(1)}
                            unit="s"
                            data-testid="missile-duration-display"
                          />
                        </div>
                      }
                    >
                      <Slider
                        data-testid="missile-duration-slider"
                        value={[settings.missileDuration]}
                        min={0.3}
                        max={5.0}
                        step={0.1}
                        onValueChange={(v) =>
                          setSettings((s) => ({ ...s, missileDuration: v[0] ?? s.missileDuration }))
                        }
                      />
                    </ControlGroup>

                    <ControlGroup
                      title="Auto Rotation"
                      description="Toggle automatic planet rotation. Use scroll to zoom in/out."
                      right={
                        <div className="flex items-center gap-2">
                          <RotateCw className="size-4 text-primary" />
                          <Switch
                            data-testid="rotation-toggle"
                            checked={autoRotate}
                            onCheckedChange={onToggleRotation}
                          />
                        </div>
                      }
                    >
                      <div className="flex gap-2">
                        <GlowButton
                          data-testid="toggle-rotation-button"
                          variant="secondary"
                          className="w-full"
                          onClick={onToggleRotation}
                        >
                          <RotateCw className="size-4" />
                          {autoRotate ? "Stop rotation" : "Start rotation"}
                        </GlowButton>
                      </div>
                    </ControlGroup>
                  </SidebarGroupContent>
                </SidebarGroup>

                <div className="my-5 hairline opacity-70" />

                <SidebarGroup>
                  <SidebarGroupLabel className="text-xs tracking-wide text-muted-foreground">
                    Actions
                  </SidebarGroupLabel>
                  <SidebarGroupContent className="space-y-3">
                    <GlowButton
                      data-testid="regenerate-button"
                      variant="primary"
                      onClick={onRegenerate}
                      disabled={validateMutation.isPending}
                      className="w-full"
                    >
                      <Wand2 className="size-4" />
                      {validateMutation.isPending ? "Validating..." : "Regenerate"}
                    </GlowButton>

                    <GlowButton
                      variant="ghost"
                      className="w-full"
                      data-testid="random-seed-button"
                      onClick={() => {
                        const next = `seed-${Math.floor(Math.random() * 9999)
                          .toString()
                          .padStart(4, "0")}`;
                        setSettings((s) => ({ ...s, seed: next }));
                        toast({ title: "Seed randomized", description: next });
                      }}
                    >
                      <RefreshCcw className="size-4" />
                      Random seed
                    </GlowButton>

                    <GlowButton
                      variant="ghost"
                      className="w-full"
                      data-testid="reset-button"
                      onClick={() => {
                        setSettings(DEFAULT_SETTINGS);
                        rendererRef.current?.setSettings(DEFAULT_SETTINGS, { rebuild: true, redistribute: true });
                        toast({ title: "Reset", description: "Back to the default studio preset." });
                      }}
                    >
                      Reset preset
                    </GlowButton>
                  </SidebarGroupContent>
                </SidebarGroup>
              </SidebarContent>

              <SidebarFooter className="p-5">
                <div
                  className="
                    rounded-2xl border border-white/10 bg-black/25 p-3
                    text-xs text-muted-foreground leading-relaxed noise-overlay
                  "
                >
                  Tip: right-click the planet to place{" "}
                  <span className="text-foreground font-semibold">trees</span> on land or{" "}
                  <span className="text-foreground font-semibold">boats</span> on water.
                  Press <span className="text-foreground font-semibold">M</span> to launch a{" "}
                  <span className="text-foreground font-semibold">missile</span> that creates craters!
                </div>
              </SidebarFooter>
            </Sidebar>

            <SidebarInset className="min-w-0">
              <main className="relative flex min-h-screen min-w-0 flex-1 flex-col">
                <div className="px-4 pt-4 sm:px-6 sm:pt-6 lg:px-8 lg:pt-8">
                  <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                    <div className="animate-float-in flex items-center gap-4">
                      <SidebarTrigger className="-ml-2">
                        <Menu className="size-5" />
                      </SidebarTrigger>
                      <div>
                        <h1 className="font-display text-3xl sm:text-4xl text-foreground">
                          Planet Studio
                        </h1>
                        <p className="mt-1 max-w-2xl text-sm sm:text-base text-muted-foreground">
                          Procedural terrain, shadow mapping, and ray-picked object placement — all in pure WebGL2.
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <div
                        className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/20 px-3 py-1.5 text-xs text-muted-foreground backdrop-blur"
                        data-testid="hud-fps"
                      >
                        <span className="size-2 rounded-full bg-primary shadow-[0_0_0_4px_hsl(var(--primary)/0.15)]" />
                        <span className="font-mono text-foreground/90">{hud.fps.toFixed(0)}</span>
                        <span>fps</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex-1 p-4 sm:p-6 lg:p-8">
                  <div
                    className="
                      relative h-[85vh] min-h-[600px] w-full overflow-hidden
                      rounded-3xl border border-white/10 bg-black/30
                      shadow-[0_40px_120px_-70px_rgba(0,0,0,0.95)]
                      backdrop-blur-sm noise-overlay
                    "
                  >
                    <div className="pointer-events-none absolute inset-0">
                      <div className="absolute -left-24 -top-24 size-64 rounded-full bg-primary/10 blur-3xl" />
                      <div className="absolute -right-24 -top-16 size-72 rounded-full bg-accent/10 blur-3xl" />
                      <div className="absolute -bottom-28 left-1/3 size-80 rounded-full bg-white/5 blur-3xl" />
                    </div>

                    <canvas
                      ref={canvasRef}
                      className="
                        relative h-full w-full
                        cursor-crosshair
                        focus:outline-none
                      "
                      data-testid="webgl-canvas"
                      onPointerDown={onCanvasPointerDown}
                      tabIndex={0}
                      aria-label="WebGL planet canvas"
                    />

                    <div className="pointer-events-none absolute bottom-4 left-4 right-4 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                      <div
                        className="
                          inline-flex max-w-full items-center gap-2 rounded-2xl
                          border border-white/10 bg-black/30 px-3 py-2 text-xs
                          text-muted-foreground backdrop-blur
                        "
                        data-testid="hud-seed"
                      >
                        <span className="text-foreground/80">Seed</span>
                        <span className="font-mono text-foreground">{settings.seed}</span>
                        <span className="opacity-70">•</span>
                        <span>Noise</span>
                        <span className="font-mono text-foreground">{settings.noiseType}</span>
                      </div>

                      <div
                        className="
                          inline-flex items-center gap-2 rounded-2xl
                          border border-white/10 bg-black/30 px-3 py-2 text-xs
                          text-muted-foreground backdrop-blur
                        "
                        data-testid="hud-help"
                      >
                        Right-click: <span className="text-foreground">place objects</span> •
                        M key: <span className="text-foreground">launch missile</span> •
                        Regenerate: <span className="text-foreground">rebuild mesh</span>
                      </div>
                    </div>
                  </div>
                </div>
              </main>
            </SidebarInset>
          </div>
        </SidebarProvider>
      </div>
    </>
  );
}
