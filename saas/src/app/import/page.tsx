"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Upload } from "lucide-react";
import { useState, useRef } from "react";

export default function ImportPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImport = async () => {
    if (!file) return;
    setStatus("Importing...");
    // TODO: Upload CSV via tRPC or API route
    setStatus(`Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB). Upload not yet connected.`);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Import Inventory</h2>

      <Card className="max-w-lg">
        <CardHeader>
          <CardTitle className="text-base">Upload TCGPlayer Export</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Export your inventory from TCGPlayer as a CSV file, then upload it here.
            Required columns: TCGplayer Id, Product Name, TCG Market Price,
            TCG Marketplace Price, Total Quantity.
          </p>

          <div
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">
              {file ? file.name : "Click to select CSV file"}
            </p>
            <Input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <Button onClick={handleImport} disabled={!file} className="w-full">
            Import
          </Button>

          {status && (
            <p className="text-sm text-muted-foreground">{status}</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
