"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";

export default function SettingsPage() {
  const [trainStatus, setTrainStatus] = useState("");
  const [predictStatus, setPredictStatus] = useState("");
  const [backtestStatus, setBacktestStatus] = useState("");

  const triggerAction = async (
    action: string,
    setStatus: (s: string) => void,
  ) => {
    setStatus(`${action}...`);
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_ML_SERVICE_URL ?? "http://localhost:8000"}/${action}`,
        { method: "POST" },
      );
      if (res.ok) {
        const data = await res.json();
        setStatus(`Done: ${JSON.stringify(data).slice(0, 200)}`);
      } else {
        setStatus(`Error: ${res.status} ${res.statusText}`);
      }
    } catch (e) {
      setStatus(`Failed to connect to ML service. Is it running?`);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Settings</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">ML Pipeline</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Train Model</p>
                <p className="text-xs text-muted-foreground">
                  Retrain XGBoost spike classifier
                </p>
              </div>
              <Button
                size="sm"
                onClick={() => triggerAction("train", setTrainStatus)}
              >
                Train
              </Button>
            </div>
            {trainStatus && (
              <p className="text-xs text-muted-foreground">{trainStatus}</p>
            )}

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Run Predictions</p>
                <p className="text-xs text-muted-foreground">
                  Score all inventory cards
                </p>
              </div>
              <Button
                size="sm"
                onClick={() => triggerAction("predict", setPredictStatus)}
              >
                Predict
              </Button>
            </div>
            {predictStatus && (
              <p className="text-xs text-muted-foreground">{predictStatus}</p>
            )}

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Backtest</p>
                <p className="text-xs text-muted-foreground">
                  Evaluate model on historical data
                </p>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={() => triggerAction("backtest", setBacktestStatus)}
              >
                Backtest
              </Button>
            </div>
            {backtestStatus && (
              <p className="text-xs text-muted-foreground">{backtestStatus}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">System Info</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Database</span>
              <Badge variant="outline">PostgreSQL + TimescaleDB</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">ML Service</span>
              <Badge variant="outline">FastAPI + XGBoost</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Feature Count</span>
              <span className="font-medium">28</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Spike Threshold</span>
              <span className="font-medium">&gt;20% in 30 days</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">HOLD Threshold</span>
              <span className="font-medium">&ge;60% probability</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
