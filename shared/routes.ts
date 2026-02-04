import { z } from "zod";
import { planetSettingsSchema } from "./schema";

export const errorSchemas = {
  validation: z.object({
    message: z.string(),
    field: z.string().optional(),
  }),
  internal: z.object({
    message: z.string(),
  }),
};

export const api = {
  planet: {
    validateSettings: {
      method: "POST" as const,
      path: "/api/planet/validate-settings",
      input: planetSettingsSchema,
      responses: {
        200: planetSettingsSchema,
        400: errorSchemas.validation,
      },
    },
  },
};

export function buildUrl(
  path: string,
  params?: Record<string, string | number>
): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}

export type PlanetSettingsInput = z.infer<typeof api.planet.validateSettings.input>;
export type PlanetSettingsResponse = z.infer<
  typeof api.planet.validateSettings.responses[200]
>;
export type ValidationError = z.infer<typeof errorSchemas.validation>;
