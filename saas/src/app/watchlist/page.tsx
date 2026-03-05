"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type WatchlistItem = {
  tcgplayerId: string;
  productName: string;
  currentPrice: number;
  spikeProb: number;
  trend: string;
};

const PLACEHOLDER: WatchlistItem[] = [];

export default function WatchlistPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Watchlist</h2>
        <Badge variant="outline" className="border-orange-500 text-orange-600">
          {PLACEHOLDER.length} HOLD signals
        </Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Cards with spike probability &ge; 60%
          </CardTitle>
        </CardHeader>
        <CardContent>
          {PLACEHOLDER.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">
              No cards on the watchlist. Run predictions to identify potential spikes.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Card</TableHead>
                  <TableHead className="text-right">Current Price</TableHead>
                  <TableHead className="text-right">Spike Probability</TableHead>
                  <TableHead>Trend</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {PLACEHOLDER.map((item) => (
                  <TableRow key={item.tcgplayerId}>
                    <TableCell className="font-medium">
                      {item.productName}
                    </TableCell>
                    <TableCell className="text-right">
                      ${item.currentPrice.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right">
                      <span className="text-orange-600 font-medium">
                        {(item.spikeProb * 100).toFixed(1)}%
                      </span>
                    </TableCell>
                    <TableCell>
                      {item.trend === "up" ? "↑" : item.trend === "down" ? "↓" : "→"}{" "}
                      {item.trend}
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
