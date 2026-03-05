import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { router, publicProcedure } from "../trpc";

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL ?? "http://localhost:8000";

async function callMlService(
  path: string,
  method: "GET" | "POST" = "POST",
  body?: Record<string, unknown>
): Promise<unknown> {
  const url = `${ML_SERVICE_URL}${path}`;
  try {
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: `ML service returned ${res.status}: ${text}`,
      });
    }

    return res.json();
  } catch (err) {
    if (err instanceof TRPCError) throw err;
    throw new TRPCError({
      code: "INTERNAL_SERVER_ERROR",
      message: `Failed to reach ML service at ${url}: ${(err as Error).message}`,
    });
  }
}

export const mlRouter = router({
  status: publicProcedure.query(async ({ ctx }) => {
    const model = await ctx.prisma.model.findFirst({
      orderBy: { trainedAt: "desc" },
      select: {
        id: true,
        trainedAt: true,
        numSamples: true,
        spikeRate: true,
        featureCols: true,
        valAccuracy: true,
        valAuc: true,
        valPrecision: true,
        valRecall: true,
        hyperparameters: true,
        featureImportance: true,
      },
    });

    return model;
  }),

  triggerTrain: publicProcedure.mutation(async () => {
    const result = await callMlService("/train", "POST");
    return result;
  }),

  triggerPredict: publicProcedure
    .input(z.object({ userId: z.string() }))
    .mutation(async ({ input }) => {
      const result = await callMlService("/predict", "POST", {
        userId: input.userId,
      });
      return result;
    }),

  triggerBacktest: publicProcedure.mutation(async () => {
    const result = await callMlService("/backtest", "POST");
    return result;
  }),
});
