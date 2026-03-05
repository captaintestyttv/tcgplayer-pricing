import { router } from "../trpc";
import { cardsRouter } from "./cards";
import { pricesRouter } from "./prices";
import { predictionsRouter } from "./predictions";
import { inventoryRouter } from "./inventory";
import { mlRouter } from "./ml";

export const appRouter = router({
  cards: cardsRouter,
  prices: pricesRouter,
  predictions: predictionsRouter,
  inventory: inventoryRouter,
  ml: mlRouter,
});

export type AppRouter = typeof appRouter;
