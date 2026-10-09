import { useEffect, useState } from "react";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { api } from "@/api";
import { usePolling } from "@/lib/use-polling";
import { cn } from "@/lib/utils";

interface HealthData {
  status: string;
  uc_catalog: string;
  uc_schema: string;
  volume_name: string;
  volume_path: string;
  extracted_frames_dir: string;
  eval_frames_dir: string;
  mlflow_experiment_path?: string;
}

export function HealthIndicator() {
  const [open, setOpen] = useState(false);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchHealth = async () => {
    try {
      const data = await api.health();
      setHealth(data);
      setError(null);
    } catch (e) {
      setError(e as Error);
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  // Retry with backoff on failure
  usePolling(
    async () => {
      try {
        const data = await api.health();
        setHealth(data);
        setError(null);
      } catch (e) {
        setError(e as Error);
      }
    },
    30000, // 30s interval
    () => error === null // stop polling when healthy
  );

  const isHealthy = error === null && health !== null;
  const status = loading ? "checking" : isHealthy ? "healthy" : "error";

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          className={cn(
            "inline-flex items-center justify-center size-2.5 rounded-full transition-colors",
            status === "healthy" && "bg-emerald-500",
            status === "error" && "bg-red-500",
            status === "checking" && "bg-amber-500 animate-pulse"
          )}
          title={
            status === "healthy"
              ? "Backend healthy"
              : status === "error"
                ? "Backend error — click for details"
                : "Checking backend…"
          }
        />
      </PopoverTrigger>
      {open && (
        <PopoverContent className="relative right-0">
          <div className="space-y-2 text-xs">
            {isHealthy && health ? (
              <>
                <div className="flex items-center gap-2 pb-2 border-b">
                  <div className="size-2 rounded-full bg-emerald-500" />
                  <span className="font-medium">Backend healthy</span>
                </div>
                <div className="grid grid-cols-[80px_1fr] gap-1.5">
                  <span className="text-muted-foreground">Catalog:</span>
                  <code className="font-mono text-[10px] break-all">{health.uc_catalog}</code>
                  <span className="text-muted-foreground">Schema:</span>
                  <code className="font-mono text-[10px] break-all">{health.uc_schema}</code>
                  <span className="text-muted-foreground">Volume:</span>
                  <code className="font-mono text-[10px] break-all">{health.volume_path}</code>
                  {health.mlflow_experiment_path && (
                    <>
                      <span className="text-muted-foreground">MLflow:</span>
                      <code className="font-mono text-[10px] break-all">{health.mlflow_experiment_path}</code>
                    </>
                  )}
                </div>
              </>
            ) : error ? (
              <>
                <div className="flex items-center gap-2 pb-2 border-b">
                  <div className="size-2 rounded-full bg-red-500" />
                  <span className="font-medium text-destructive">Backend error</span>
                </div>
                <p className="text-muted-foreground">{error.message}</p>
                <button
                  onClick={() => {
                    setLoading(true);
                    fetchHealth();
                  }}
                  className="mt-2 text-xs text-primary hover:underline"
                >
                  Retry
                </button>
              </>
            ) : (
              <p className="text-muted-foreground">Checking…</p>
            )}
          </div>
        </PopoverContent>
      )}
    </Popover>
  );
}
