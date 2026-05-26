import { useEffect, useState } from "react";
import {
  CheckCircle2, ChevronDown, ChevronRight, ExternalLink, Info, Loader2,
  RefreshCcw, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/api";
import { cn } from "@/lib/utils";

type Check = {
  name: string;
  ok: boolean;
  detail: string;
  remediation: string | null;
  docs_url: string | null;
};

export default function Setup() {
  const [checks, setChecks] = useState<Check[]>([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<{ n_ok: number; n_total: number; ready: boolean } | null>(null);

  const refresh = () => {
    setLoading(true);
    setError(null);
    api.setupCheck()
      .then((r) => {
        setChecks(r.checks);
        setSummary({ n_ok: r.n_ok, n_total: r.n_total, ready: r.ready });
      })
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false));
  };
  useEffect(refresh, []);

  const toggle = (name: string) => {
    setExpanded((cur) => {
      const next = new Set(cur);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Setup</h1>
        <p className="text-sm text-muted-foreground">
          Preflight checks for every dependency the workbench needs. Run after a fresh deploy,
          when something's broken, or any time you want a quick health snapshot. Each failed
          check expands to show the exact remediation.
        </p>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
          <div>
            <CardTitle className="text-base">Health</CardTitle>
            <CardDescription>
              {summary
                ? `${summary.n_ok}/${summary.n_total} checks passing${summary.ready ? " — all green" : ""}`
                : "—"}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {summary && (
              <Badge
                variant={summary.ready ? "default" : "destructive"}
                className="gap-1 font-normal"
              >
                {summary.ready
                  ? <><CheckCircle2 className="size-3" /> ready</>
                  : <><XCircle className="size-3" /> {summary.n_total - summary.n_ok} failing</>}
              </Badge>
            )}
            <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
              {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCcw className="size-3.5" />}
              Re-check
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {error && (
            <p className="px-4 py-3 text-sm text-destructive">{error}</p>
          )}
          <div className="divide-y">
            {checks.map((c) => {
              const isOpen = expanded.has(c.name);
              return (
                <div key={c.name} className="px-4 py-3">
                  <button
                    onClick={() => c.remediation && toggle(c.name)}
                    className={cn(
                      "flex w-full items-start gap-3 text-left",
                      c.remediation && !c.ok && "cursor-pointer",
                    )}
                  >
                    {c.ok
                      ? <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-500" />
                      : <XCircle className="mt-0.5 size-4 shrink-0 text-destructive" />}
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium">{c.name}</span>
                        {!c.ok && c.remediation && (
                          isOpen ? <ChevronDown className="size-3.5 text-muted-foreground" /> : <ChevronRight className="size-3.5 text-muted-foreground" />
                        )}
                      </div>
                      <p className={cn(
                        "mt-0.5 text-xs",
                        c.ok ? "text-muted-foreground" : "text-destructive",
                      )}>
                        {c.detail || "—"}
                      </p>
                    </div>
                  </button>
                  {isOpen && c.remediation && (
                    <div className="ml-7 mt-2 rounded-md border bg-muted/40 p-3 text-xs">
                      <div className="mb-1 flex items-center gap-1 font-medium text-foreground">
                        <Info className="size-3.5" /> How to fix
                      </div>
                      <p className="whitespace-pre-wrap leading-relaxed text-muted-foreground">
                        {c.remediation}
                      </p>
                      {c.docs_url && (
                        <a
                          href={c.docs_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="mt-2 flex items-center gap-1 text-primary hover:underline"
                        >
                          <ExternalLink className="size-3" /> Docs
                        </a>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
            {checks.length === 0 && !loading && !error && (
              <p className="px-4 py-3 text-xs text-muted-foreground">No checks yet.</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Onboarding playbook</CardTitle>
          <CardDescription>The opinionated sequence for setting up the workbench in a fresh workspace.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <Step n={1} title="Deploy the bundle">
            <code className="block rounded bg-muted px-2 py-1 text-xs">
              cp module.env.sample module.env  # then edit catalog / schema / warehouse / lakebase / hf scope
              <br/>./deploy.sh aws  # or azure / gcp / dev
            </code>
            <p className="text-xs text-muted-foreground">
              Creates the App, UC schema/volume, workbench secret scope, and runs a post-deploy job
              that grants the App SP every UC privilege it needs and creates the Lakebase role.
            </p>
          </Step>
          <Step n={2} title="Drop your data in the Volume">
            <p className="text-xs text-muted-foreground">
              MP4s into <code>/Volumes/&lt;catalog&gt;/&lt;schema&gt;/&lt;volume&gt;/videos/inbox/</code>,
              or JPGs into <code>/Volumes/.../images/&lt;batch_name&gt;/</code>. The <strong>Library</strong>
              tab will detect them within a few seconds.
            </p>
          </Step>
          <Step n={3} title="Cache local models">
            <p className="text-xs text-muted-foreground">
              From the <strong>Library</strong> tab, click ingest. From outside the app: run the
              <code> setup_cache</code> notebook (with your HF token in the configured secret scope) to
              snapshot Qwen3-VL-8B + MedGemma weights into the Volume. Both models then become
              selectable in <strong>Playground</strong>'s Local section.
            </p>
          </Step>
          <Step n={4} title="Label, optimize, train, deploy">
            <p className="text-xs text-muted-foreground">
              The standard workflow: Playground → Label → Optimize prompt → Train fine-tune → Deploy.
              Every step writes to MLflow and registers artifacts in Unity Catalog, so the whole loop
              is reproducible.
            </p>
          </Step>
        </CardContent>
      </Card>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full border bg-card text-xs font-medium">
        {n}
      </div>
      <div className="min-w-0 flex-1">
        <p className="font-medium">{title}</p>
        <div className="mt-1 space-y-2">{children}</div>
      </div>
    </div>
  );
}
