"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type BacktestMetrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc: number;
  confusion: { tp: number; fp: number; fn: number; tn: number };
  calibration: Array<{ bin: string; predicted: number; actual: number; count: number }>;
};

export default function BacktestPage() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Backtest Results</h2>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <MetricCard label="Accuracy" value="—" />
        <MetricCard label="Precision" value="—" />
        <MetricCard label="Recall" value="—" />
        <MetricCard label="F1 Score" value="—" />
        <MetricCard label="AUC" value="—" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Confusion Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2 max-w-xs">
              <ConfusionCell label="TP" value="—" className="bg-green-50 dark:bg-green-950" />
              <ConfusionCell label="FP" value="—" className="bg-red-50 dark:bg-red-950" />
              <ConfusionCell label="FN" value="—" className="bg-red-50 dark:bg-red-950" />
              <ConfusionCell label="TN" value="—" className="bg-green-50 dark:bg-green-950" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Probability Calibration</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground py-8 text-center">
              Run backtest to view calibration data.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="text-2xl font-bold text-center">{value}</div>
        <p className="text-xs text-muted-foreground text-center mt-1">{label}</p>
      </CardContent>
    </Card>
  );
}

function ConfusionCell({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className: string;
}) {
  return (
    <div className={`p-4 rounded-md text-center ${className}`}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-xl font-bold mt-1">{value}</div>
    </div>
  );
}
