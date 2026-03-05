import { z } from "zod";
import { router, publicProcedure } from "../trpc";

export const pricesRouter = router({
  getHistory: publicProcedure
    .input(
      z.object({
        cardUuid: z.string(),
        priceType: z.string().optional(),
        days: z.number().int().min(1).max(365).optional().default(90),
      })
    )
    .query(async ({ ctx, input }) => {
      const since = new Date();
      since.setDate(since.getDate() - input.days);

      return ctx.prisma.priceHistory.findMany({
        where: {
          cardUuid: input.cardUuid,
          date: { gte: since },
          ...(input.priceType ? { priceType: input.priceType } : {}),
        },
        orderBy: { date: "asc" },
      });
    }),

  getLatest: publicProcedure
    .input(z.object({ cardUuid: z.string() }))
    .query(async ({ ctx, input }) => {
      // Get the latest price for each price_type by finding max date per type
      const priceTypes = await ctx.prisma.priceHistory.findMany({
        where: { cardUuid: input.cardUuid },
        distinct: ["priceType"],
        orderBy: { date: "desc" },
        select: { priceType: true },
      });

      const latestPrices = await Promise.all(
        priceTypes.map((pt) =>
          ctx.prisma.priceHistory.findFirst({
            where: {
              cardUuid: input.cardUuid,
              priceType: pt.priceType,
            },
            orderBy: { date: "desc" },
          })
        )
      );

      return latestPrices.filter(Boolean);
    }),

  getBulk: publicProcedure
    .input(
      z.object({
        cardUuids: z.array(z.string()).min(1).max(500),
      })
    )
    .query(async ({ ctx, input }) => {
      // For each card, get the latest price per price_type
      const results = await Promise.all(
        input.cardUuids.map(async (cardUuid) => {
          const priceTypes = await ctx.prisma.priceHistory.findMany({
            where: { cardUuid },
            distinct: ["priceType"],
            orderBy: { date: "desc" },
            select: { priceType: true },
          });

          const latestPrices = await Promise.all(
            priceTypes.map((pt) =>
              ctx.prisma.priceHistory.findFirst({
                where: { cardUuid, priceType: pt.priceType },
                orderBy: { date: "desc" },
              })
            )
          );

          return {
            cardUuid,
            prices: latestPrices.filter(Boolean),
          };
        })
      );

      return results;
    }),
});
