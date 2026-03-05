/**
 * Data migration script: reads JSON files from the Python tool's data directory
 * and inserts them into the PostgreSQL database via Prisma.
 *
 * Usage: npx tsx scripts/migrate-data.ts
 */

import { PrismaClient } from "../src/generated/prisma/client";
import * as fs from "fs";
import * as path from "path";

const prisma = new PrismaClient();

// Project root is one level above saas/
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const DATA_DIR = path.join(PROJECT_ROOT, "data", "mtgjson");
const MODELS_DIR = path.join(PROJECT_ROOT, "models");

const TRAINING_CARDS_PATH = path.join(DATA_DIR, "training_cards.json");
const INVENTORY_CARDS_PATH = path.join(DATA_DIR, "inventory_cards.json");
const MODEL_META_PATH = path.join(MODELS_DIR, "spike_classifier_meta.json");
const MODEL_BLOB_PATH = path.join(MODELS_DIR, "spike_classifier.json");

// Batch size for createMany operations
const BATCH_SIZE = 1000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fileExists(p: string): boolean {
  try {
    fs.accessSync(p, fs.constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

function readJson(p: string): unknown | null {
  if (!fileExists(p)) {
    console.log(`  [SKIP] File not found: ${p}`);
    return null;
  }
  const raw = fs.readFileSync(p, "utf-8");
  return JSON.parse(raw);
}

function progress(label: string, current: number, total: number) {
  const pct = Math.round((current / total) * 100);
  const bar = "=".repeat(Math.floor(pct / 2)).padEnd(50, " ");
  process.stdout.write(`\r  [${bar}] ${pct}% (${current}/${total}) ${label}`);
  if (current === total) process.stdout.write("\n");
}

function parseDateSafe(dateStr: string | null | undefined): Date | null {
  if (!dateStr) return null;
  const d = new Date(dateStr);
  return isNaN(d.getTime()) ? null : d;
}

// ---------------------------------------------------------------------------
// Step 1: Migrate cards from training_cards.json
// ---------------------------------------------------------------------------
async function migrateCards(
  trainingCards: Record<string, Record<string, unknown>>
): Promise<number> {
  console.log("\n--- Step 1: Migrating cards ---");
  const uuids = Object.keys(trainingCards);
  const total = uuids.length;
  console.log(`  Found ${total} cards in training_cards.json`);

  let inserted = 0;

  for (let i = 0; i < total; i += BATCH_SIZE) {
    const batch = uuids.slice(i, i + BATCH_SIZE);
    const records = batch.map((uuid) => {
      const card = trainingCards[uuid];
      return {
        uuid,
        name: (card.name as string) || "Unknown",
        setCode: (card.setCode as string) || null,
        rarity: (card.rarity as string) || null,
        manaValue: typeof card.manaValue === "number" ? card.manaValue : null,
        colorIdentity: (card.colorIdentity as string[]) || [],
        types: (card.types as string[]) || [],
        subtypes: (card.subtypes as string[]) || [],
        supertypes: (card.supertypes as string[]) || [],
        keywords: (card.keywords as string[]) || [],
        legalities: (card.legalities as Record<string, string>) || {},
        edhrecRank:
          typeof card.edhrecRank === "number" ? card.edhrecRank : null,
        edhrecSaltiness:
          typeof card.edhrecSaltiness === "number"
            ? card.edhrecSaltiness
            : null,
        isReserved: Boolean(card.isReserved),
        printings: (card.printings as string[]) || [],
        setReleaseDate: parseDateSafe(card.setReleaseDate as string),
      };
    });

    await prisma.card.createMany({
      data: records,
      skipDuplicates: true,
    });

    inserted += batch.length;
    progress("cards", Math.min(inserted, total), total);
  }

  return inserted;
}

// ---------------------------------------------------------------------------
// Step 2: Migrate TCGPlayer ID mappings from inventory_cards.json
// ---------------------------------------------------------------------------
async function migrateTcgplayerIds(
  inventoryCards: Record<string, Record<string, unknown>>,
  validUuids: Set<string>
): Promise<number> {
  console.log("\n--- Step 2: Migrating TCGPlayer ID mappings ---");
  const skuIds = Object.keys(inventoryCards);
  const total = skuIds.length;
  console.log(`  Found ${total} inventory entries`);

  let inserted = 0;
  let skipped = 0;

  for (let i = 0; i < total; i += BATCH_SIZE) {
    const batch = skuIds.slice(i, i + BATCH_SIZE);
    const records: { tcgplayerId: string; cardUuid: string; skuId: string }[] =
      [];

    for (const skuId of batch) {
      const card = inventoryCards[skuId];
      const uuid = card.uuid as string;
      if (!uuid || !validUuids.has(uuid)) {
        skipped++;
        continue;
      }
      records.push({
        tcgplayerId: skuId,
        cardUuid: uuid,
        skuId: skuId,
      });
    }

    if (records.length > 0) {
      await prisma.cardTcgplayerId.createMany({
        data: records,
        skipDuplicates: true,
      });
    }

    inserted += records.length;
    progress(
      "tcgplayer_ids",
      Math.min(i + batch.length, total),
      total
    );
  }

  console.log(`  Inserted: ${inserted}, Skipped (no matching card): ${skipped}`);
  return inserted;
}

// ---------------------------------------------------------------------------
// Step 3: Migrate price history from training_cards.json
// ---------------------------------------------------------------------------
async function migratePriceHistory(
  trainingCards: Record<string, Record<string, unknown>>
): Promise<number> {
  console.log("\n--- Step 3: Migrating price history ---");
  const uuids = Object.keys(trainingCards);
  const total = uuids.length;

  let totalPrices = 0;
  let cardsDone = 0;

  for (const uuid of uuids) {
    const card = trainingCards[uuid];
    const priceEntries: {
      cardUuid: string;
      date: Date;
      priceType: string;
      source: string;
      price: number;
    }[] = [];

    // Collect all three price types
    const priceTypes: [string, string][] = [
      ["price_history", "normal"],
      ["foil_price_history", "foil"],
      ["buylist_price_history", "buylist"],
    ];

    for (const [field, priceType] of priceTypes) {
      const history = card[field] as Record<string, number> | undefined;
      if (!history) continue;

      for (const [dateStr, price] of Object.entries(history)) {
        if (typeof price !== "number" || isNaN(price)) continue;
        const date = parseDateSafe(dateStr);
        if (!date) continue;

        priceEntries.push({
          cardUuid: uuid,
          date,
          priceType,
          source: "mtgjson",
          price,
        });
      }
    }

    // Batch insert price entries for this card
    if (priceEntries.length > 0) {
      for (let j = 0; j < priceEntries.length; j += BATCH_SIZE) {
        const batch = priceEntries.slice(j, j + BATCH_SIZE);
        await prisma.priceHistory.createMany({
          data: batch,
          skipDuplicates: true,
        });
      }
      totalPrices += priceEntries.length;
    }

    cardsDone++;
    if (cardsDone % 500 === 0 || cardsDone === total) {
      progress("price_history", cardsDone, total);
    }
  }

  console.log(`  Total price records inserted: ${totalPrices}`);
  return totalPrices;
}

// ---------------------------------------------------------------------------
// Step 4: Migrate model metadata
// ---------------------------------------------------------------------------
async function migrateModel(): Promise<boolean> {
  console.log("\n--- Step 4: Migrating model metadata ---");

  const meta = readJson(MODEL_META_PATH) as Record<string, unknown> | null;
  if (!meta) return false;

  let modelBlob: Buffer | null = null;
  if (fileExists(MODEL_BLOB_PATH)) {
    const raw = fs.readFileSync(MODEL_BLOB_PATH);
    modelBlob = Buffer.from(raw);
    console.log(
      `  Model blob size: ${(modelBlob.length / 1024).toFixed(1)} KB`
    );
  }

  const trainedAt = parseDateSafe(meta.trained_at as string) || new Date();
  const metrics = meta.validation_metrics as Record<string, number> | undefined;
  const hyperparams = meta.hyperparameters as Record<string, unknown> | undefined;
  const featureImportance = meta.feature_importance as
    | Record<string, number>
    | undefined;

  await prisma.model.create({
    data: {
      trainedAt,
      numSamples: (meta.num_samples as number) || 0,
      spikeRate: (meta.spike_rate as number) || 0,
      featureCols: (meta.feature_cols as string[]) || [],
      valAccuracy: metrics?.accuracy ?? null,
      valAuc: metrics?.auc ?? null,
      valPrecision: metrics?.precision ?? null,
      valRecall: metrics?.recall ?? null,
      hyperparameters: hyperparams || undefined,
      featureImportance: featureImportance || undefined,
      modelBlob: modelBlob,
    },
  });

  console.log("  Model metadata inserted successfully");
  return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log("=== TCGPlayer Data Migration ===");
  console.log(`Project root: ${PROJECT_ROOT}`);
  console.log(`Data dir:     ${DATA_DIR}`);

  const startTime = Date.now();

  // Load training cards
  const trainingCards = readJson(TRAINING_CARDS_PATH) as Record<
    string,
    Record<string, unknown>
  > | null;

  if (!trainingCards) {
    console.error(
      "\nERROR: training_cards.json not found. Run 'bash scripts/monitor.sh sync' first."
    );
    process.exit(1);
  }

  // Step 1: Cards
  await migrateCards(trainingCards);

  // Step 2: TCGPlayer ID mappings
  const inventoryCards = readJson(INVENTORY_CARDS_PATH) as Record<
    string,
    Record<string, unknown>
  > | null;

  if (inventoryCards) {
    const validUuids = new Set(Object.keys(trainingCards));
    await migrateTcgplayerIds(inventoryCards, validUuids);
  } else {
    console.log("\n--- Step 2: SKIPPED (inventory_cards.json not found) ---");
  }

  // Step 3: Price history
  await migratePriceHistory(trainingCards);

  // Step 4: Model
  await migrateModel();

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n=== Migration complete in ${elapsed}s ===`);
}

main()
  .catch((err) => {
    console.error("\nMigration failed:", err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
