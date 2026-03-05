import { z } from "zod";
import { router, publicProcedure } from "../trpc";

export const cardsRouter = router({
  search: publicProcedure
    .input(
      z.object({
        query: z.string().min(1),
        limit: z.number().int().min(1).max(100).optional().default(20),
      })
    )
    .query(async ({ ctx, input }) => {
      return ctx.prisma.card.findMany({
        where: {
          name: {
            contains: input.query,
            mode: "insensitive",
          },
        },
        select: {
          uuid: true,
          name: true,
          setCode: true,
          rarity: true,
        },
        take: input.limit,
        orderBy: { name: "asc" },
      });
    }),

  getById: publicProcedure
    .input(z.object({ uuid: z.string() }))
    .query(async ({ ctx, input }) => {
      const card = await ctx.prisma.card.findUnique({
        where: { uuid: input.uuid },
        include: {
          tcgplayerIds: true,
        },
      });
      if (!card) {
        return null;
      }
      return card;
    }),

  list: publicProcedure
    .input(
      z.object({
        page: z.number().int().min(1).optional().default(1),
        pageSize: z.number().int().min(1).max(100).optional().default(50),
        setCode: z.string().optional(),
      })
    )
    .query(async ({ ctx, input }) => {
      const where = input.setCode ? { setCode: input.setCode } : {};
      const [items, total] = await Promise.all([
        ctx.prisma.card.findMany({
          where,
          select: {
            uuid: true,
            name: true,
            setCode: true,
            rarity: true,
            manaValue: true,
            colorIdentity: true,
            types: true,
          },
          skip: (input.page - 1) * input.pageSize,
          take: input.pageSize,
          orderBy: { name: "asc" },
        }),
        ctx.prisma.card.count({ where }),
      ]);

      return {
        items,
        total,
        page: input.page,
        pageSize: input.pageSize,
        totalPages: Math.ceil(total / input.pageSize),
      };
    }),
});
