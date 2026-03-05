"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useParams } from "next/navigation";
import { PriceChart } from "@/components/price-chart";

export default function CardDetailPage() {
  const params = useParams();
  const uuid = params.uuid as string;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <h2 className="text-2xl font-bold">Card Detail</h2>
        <Badge variant="outline" className="font-mono text-xs">
          {uuid}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Price History</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="normal">
                <TabsList>
                  <TabsTrigger value="normal">Normal</TabsTrigger>
                  <TabsTrigger value="foil">Foil</TabsTrigger>
                  <TabsTrigger value="buylist">Buylist</TabsTrigger>
                </TabsList>
                <TabsContent value="normal" className="pt-4">
                  <PriceChart data={[]} />
                </TabsContent>
                <TabsContent value="foil" className="pt-4">
                  <PriceChart data={[]} />
                </TabsContent>
                <TabsContent value="buylist" className="pt-4">
                  <PriceChart data={[]} />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Card Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <InfoRow label="Name" value="—" />
              <InfoRow label="Set" value="—" />
              <InfoRow label="Rarity" value="—" />
              <InfoRow label="Mana Value" value="—" />
              <InfoRow label="Colors" value="—" />
              <InfoRow label="Types" value="—" />
              <InfoRow label="EDHREC Rank" value="—" />
              <InfoRow label="Reserved List" value="—" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Prediction</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <InfoRow label="Spike Probability" value="—" />
              <InfoRow label="Signal" value="—" />
              <InfoRow label="Trend" value="—" />
              <InfoRow label="Predicted 7d" value="—" />
              <InfoRow label="Predicted 30d" value="—" />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
