import { initTRPC, TRPCError } from "@trpc/server";
import superjson from "superjson";
import { prisma } from "./db";

export type Context = {
  prisma: typeof prisma;
  session: { userId: string; user: { name?: string; email?: string } } | null;
};

export async function createContext(): Promise<Context> {
  // In a full setup this would extract the session from the request headers
  // via NextAuth getServerSession. For now we pass prisma and a null session.
  return {
    prisma,
    session: null,
  };
}

const t = initTRPC.context<Context>().create({
  transformer: superjson,
});

export const router = t.router;
export const publicProcedure = t.procedure;

export const protectedProcedure = t.procedure.use(async ({ ctx, next }) => {
  if (!ctx.session) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "You must be logged in to access this resource.",
    });
  }
  return next({
    ctx: {
      ...ctx,
      session: ctx.session,
    },
  });
});
