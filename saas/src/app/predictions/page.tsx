"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useState } from "react";

type Prediction = {
  tcgplayerId: string;
  productName: string;
  currentPrice: number;
  marketPrice: number;
  suggestedPrice: number;
  action: string;
  signal: string;
  spikeProb: number;
  trend: string;
  predicted7d: number | null;
  predicted30d: number | null;
};

// Placeholder data — will be replaced with tRPC query
const PLACEHOLDER: Prediction[] = [];

export default function PredictionsPage() {
  const [search, setSearch] = useState("");
  const [signalFilter, setSignalFilter] = useState<string>("all");

  const filtered = PLACEHOLDER.filter((p) => {
    const matchSearch = p.productName
      .toLowerCase()
      .includes(search.toLowerCase());
    const matchSignal =
      signalFilter === "all" || p.signal === signalFilter;
    return matchSearch && matchSignal;
  });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Predictions</h2>

      <div className="flex gap-4">
        <Input
          placeholder="Search cards..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm"
        />
        <div className="flex gap-2">
          {["all", "HOLD", "SELL_NOW"].map((s) => (
            <button
              key={s}
              onClick={() => setSignalFilter(s)}
              className={`px-3 py-1 text-sm rounded-md border transition-colors ${
                signalFilter === s
                  ? "bg-primary text-primary-foreground"
                  : "bg-background hover:bg-accent"
              }`}
            >
              {s === "all" ? "All" : s}
            </button>
          ))}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {filtered.length} predictions
          </CardTitle>
        </CardHeader>
        <CardContent>
          {filtered.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">
              No predictions available. Run the predict pipeline to generate results.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Card</TableHead>
                  <TableHead className="text-right">Current</TableHead>
                  <TableHead className="text-right">Market</TableHead>
                  <TableHead className="text-right">Suggested</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Signal</TableHead>
                  <TableHead className="text-right">Spike %</TableHead>
                  <TableHead>Trend</TableHead>
                  <TableHead className="text-right">7d</TableHead>
                  <TableHead className="text-right">30d</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((p) => (
                  <TableRow key={p.tcgplayerId}>
                    <TableCell className="font-medium">
                      {p.productName}
                    </TableCell>
                    <TableCell className="text-right">
                      ${p.currentPrice.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right">
                      ${p.marketPrice.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right">
                      ${p.suggestedPrice.toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <ActionBadge action={p.action} />
                    </TableCell>
                    <TableCell>
                      <SignalBadge signal={p.signal} />
                    </TableCell>
                    <TableCell className="text-right">
                      {(p.spikeProb * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell>
                      <TrendIndicator trend={p.trend} />
                    </TableCell>
                    <TableCell className="text-right">
                      {p.predicted7d?.toFixed(2) ?? "—"}
                    </TableCell>
                    <TableCell className="text-right">
                      {p.predicted30d?.toFixed(2) ?? "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ActionBadge({ action }: { action: string }) {
  if (!action) return null;
  const variant = action === "RAISE" ? "default" : "destructive";
  return <Badge variant={variant}>{action}</Badge>;
}

function SignalBadge({ signal }: { signal: string }) {
  if (!signal) return null;
  return (
    <Badge
      variant="outline"
      className={
        signal === "HOLD"
          ? "border-orange-500 text-orange-600"
          : "border-red-500 text-red-600"
      }
    >
      {signal}
    </Badge>
  );
}

function TrendIndicator({ trend }: { trend: string }) {
  const color =
    trend === "up"
      ? "text-green-600"
      : trend === "down"
        ? "text-red-600"
        : "text-muted-foreground";
  const arrow = trend === "up" ? "↑" : trend === "down" ? "↓" : "→";
  return <span className={color}>{arrow} {trend}</span>;
}
