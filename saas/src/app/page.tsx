import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Eye, Package } from "lucide-react";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Dashboard</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Inventory Cards"
          value="—"
          subtitle="Total tracked"
          icon={<Package className="h-4 w-4 text-muted-foreground" />}
        />
        <StatCard
          title="HOLD Signals"
          value="—"
          subtitle="Potential spikes"
          icon={<Eye className="h-4 w-4 text-orange-500" />}
        />
        <StatCard
          title="RAISE Actions"
          value="—"
          subtitle="Underpriced cards"
          icon={<TrendingUp className="h-4 w-4 text-green-500" />}
        />
        <StatCard
          title="LOWER Actions"
          value="—"
          subtitle="Overpriced cards"
          icon={<TrendingDown className="h-4 w-4 text-red-500" />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Connect to the database to view predictions. Run{" "}
              <code className="bg-muted px-1 rounded">docker compose up</code> and{" "}
              <code className="bg-muted px-1 rounded">npx prisma migrate dev</code> to get started.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model Status</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              No model loaded. Train a model from the Settings page or run the ML service.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </CardContent>
    </Card>
  );
}
