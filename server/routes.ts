import type { Express } from "express";
import type { Server } from "http";
import { api } from "@shared/routes";
import { z } from "zod";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  app.post(api.planet.validateSettings.path, async (req, res) => {
    try {
      const settings = api.planet.validateSettings.input.parse(req.body);
      return res.status(200).json(settings);
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0]?.message ?? "Invalid request",
          field: err.errors[0]?.path?.join("."),
        });
      }

      return res.status(500).json({ message: "Internal error" });
    }
  });

  return httpServer;
}
