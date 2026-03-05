import { z } from "zod";
import { router, publicProcedure } from "../trpc";

export const predictionsRouter = router({
  list: publicProcedure
    .input(
      z.object({
        userId: z.string(),
        page: z.number().int().min(1).optional().default(1),
        pageSize: z.number().int().min(1).max(100).optional().default(50),
        signal: z.string().optional(),
      })
    )
    .query(async ({ ctx, input }) => {
      const where = {
        userId: input.userId,
        ...(input.signal ? { signal: input.signal } : {}),
      };

      const [items, total] = await Promise.all([
        ctx.prisma.prediction.findMany({
          where,
          include: {
            card: {
              select: { name: true, setCode: true, rarity: true },
            },
          },
          skip: (input.page - 1) * input.pageSize,
          take: input.pageSize,
          orderBy: { runAt: "desc" },
        }),
        ctx.prisma.prediction.count({ where }),
      ]);

      return {
        items,
        total,
        page: input.page,
        pageSize: input.pageSize,
        totalPages: Math.ceil(total / input.pageSize),
      };
    }),

  watchlist: publicProcedure
    .input(z.object({ userId: z.string() }))
    .query(async ({ ctx, input }) => {
      return ctx.prisma.prediction.findMany({
        where: {
          userId: input.userId,
          signal: "HOLD",
        },
        include: {
          card: {
            select: { name: true, setCode: true, rarity: true },
          },
        },
        orderBy: { spikeProb: "desc" },
      });
    }),

  getByCard: publicProcedure
    .input(
      z.object({
        userId: z.string(),
        tcgplayerId: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      return ctx.prisma.prediction.findMany({
        where: {
          userId: input.userId,
          tcgplayerId: input.tcgplayerId,
        },
        include: {
          card: {
            select: { name: true, setCode: true, rarity: true },
          },
        },
        orderBy: { runAt: "desc" },
      });
    }),
});
