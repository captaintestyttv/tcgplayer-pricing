import { z } from "zod";
import { router, publicProcedure } from "../trpc";

export const inventoryRouter = router({
  list: publicProcedure
    .input(z.object({ userId: z.string() }))
    .query(async ({ ctx, input }) => {
      const items = await ctx.prisma.userInventory.findMany({
        where: { userId: input.userId },
        orderBy: { importedAt: "desc" },
      });

      // Join with card names via tcgplayer ID mapping
      const tcgplayerIds = items.map((i) => i.tcgplayerId);
      const mappings = await ctx.prisma.cardTcgplayerId.findMany({
        where: { tcgplayerId: { in: tcgplayerIds } },
        include: {
          card: {
            select: { uuid: true, name: true, setCode: true, rarity: true },
          },
        },
      });

      const mappingByTcgId = new Map(
        mappings.map((m) => [m.tcgplayerId, m])
      );

      return items.map((item) => {
        const mapping = mappingByTcgId.get(item.tcgplayerId);
        return {
          ...item,
          cardUuid: mapping?.cardUuid ?? null,
          cardName: mapping?.card.name ?? null,
          setCode: mapping?.card.setCode ?? null,
          rarity: mapping?.card.rarity ?? null,
        };
      });
    }),

  import: publicProcedure
    .input(
      z.object({
        userId: z.string(),
        items: z
          .array(
            z.object({
              tcgplayerId: z.string(),
              quantity: z.number().int().min(0),
              listedPrice: z.number().min(0).optional(),
            })
          )
          .min(1)
          .max(10000),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const results = await ctx.prisma.$transaction(
        input.items.map((item) =>
          ctx.prisma.userInventory.upsert({
            where: {
              userId_tcgplayerId: {
                userId: input.userId,
                tcgplayerId: item.tcgplayerId,
              },
            },
            create: {
              userId: input.userId,
              tcgplayerId: item.tcgplayerId,
              quantity: item.quantity,
              listedPrice: item.listedPrice ?? null,
              importedAt: new Date(),
            },
            update: {
              quantity: item.quantity,
              listedPrice: item.listedPrice ?? null,
              importedAt: new Date(),
            },
          })
        )
      );

      return { imported: results.length };
    }),
});
