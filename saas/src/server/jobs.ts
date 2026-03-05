import { PgBoss } from "pg-boss";

let boss: PgBoss | null = null;

export async function getJobQueue(): Promise<PgBoss> {
  if (!boss) {
    boss = new PgBoss(process.env.DATABASE_URL!);
    await boss.start();

    await boss.work("sync-mtgjson", async (jobs) => {
      for (const job of jobs) {
        console.log(`[job:sync-mtgjson] Starting sync...`);
      }
    });

    await boss.work("import-csv", async (jobs) => {
      for (const job of jobs) {
        const data = job.data as { userId: string; filePath: string };
        console.log(`[job:import-csv] Importing for user ${data.userId}...`);
        // TODO: Parse CSV and upsert into user_inventory
      }
    });

    await boss.work("train-model", async (jobs) => {
      const mlUrl = process.env.ML_SERVICE_URL ?? "http://localhost:8000";
      console.log(`[job:train-model] Triggering training...`);
      const res = await fetch(`${mlUrl}/train`, { method: "POST" });
      if (!res.ok) throw new Error(`Train failed: ${res.status}`);
    });

    await boss.work("run-predictions", async (jobs) => {
      const mlUrl = process.env.ML_SERVICE_URL ?? "http://localhost:8000";
      for (const job of jobs) {
        const data = job.data as { userId: string };
        console.log(`[job:run-predictions] Running for user ${data.userId}...`);
        const res = await fetch(`${mlUrl}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: data.userId }),
        });
        if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
      }
    });
  }
  return boss;
}

export async function scheduleRecurringJobs() {
  const queue = await getJobQueue();
  await queue.schedule("sync-mtgjson", "0 2 * * *", {});
}

export async function enqueueJob(
  name: string,
  data: Record<string, unknown> = {},
) {
  const queue = await getJobQueue();
  return queue.send(name, data);
}
