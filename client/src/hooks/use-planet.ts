import { useMutation } from "@tanstack/react-query";
import { api } from "@shared/routes";
import type { PlanetSettingsInput, PlanetSettingsResponse, ValidationError } from "@shared/routes";
import { z } from "zod";

export function useValidatePlanetSettings() {
  return useMutation({
    mutationFn: async (input: PlanetSettingsInput): Promise<PlanetSettingsResponse> => {
      const validated = api.planet.validateSettings.input.parse(input);

      const res = await fetch(api.planet.validateSettings.path, {
        method: api.planet.validateSettings.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
        credentials: "include",
      });

      if (res.status === 400) {
        const err: ValidationError = api.planet.validateSettings.responses[400].parse(
          await res.json(),
        );
        throw new Error(err.field ? `${err.field}: ${err.message}` : err.message);
      }

      if (!res.ok) {
        throw new Error(`Failed to validate settings (${res.status})`);
      }

      const json = await res.json();
      return api.planet.validateSettings.responses[200].parse(json);
    },
    onError: (err) => {
      // Non-blocking: app can still proceed with local validation.
      if (err instanceof z.ZodError) {
        console.error("[Zod] validateSettings input invalid:", err.format());
      } else {
        console.warn("[API] validate-settings failed:", err);
      }
    },
  });
}
