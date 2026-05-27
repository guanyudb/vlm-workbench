import { useEffect, useState } from "react";
import {
  AlertCircle, CheckCircle2, Cloud, ExternalLink, Loader2, Play, RefreshCcw,
  Rocket, Trash2, Wand2, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { api, type DeployableModel, type ServingEndpointRow } from "@/api";
import { cn } from "@/lib/utils";

const WORKLOAD_TYPES = [
  { value: "CPU", label: "CPU (cheapest, no GPU)" },
  { value: "GPU_SMALL", label: "GPU_SMALL (1× T4 / cost-efficient)" },
  { value: "GPU_MEDIUM", label: "GPU_MEDIUM (1× A10G)" },
  { value: "GPU_LARGE", label: "GPU_LARGE (1× A100 — recommended for VLMs)" },
];

const WORKLOAD_SIZES = [
  { value: "Small", label: "Small (1 replica)" },
  { value: "Medium", label: "Medium (2 replicas)" },
  { value: "Large", label: "Large (4 replicas)" },
];

function StateBadge({ state, config_state }: { state: string; config_state: string | null }) {
  if (state === "READY") {
    return <Badge variant="default" className="gap-1 font-normal"><CheckCircle2 className="size-3" />ready</Badge>;
  }
  if (state === "NOT_READY" || config_state === "IN_PROGRESS") {
    return <Badge variant="secondary" className="gap-1 font-normal"><Loader2 className="size-3 animate-spin" />updating</Badge>;
  }
  if (config_state === "FAILED") {
    return <Badge variant="destructive" className="gap-1 font-normal"><XCircle className="size-3" />failed</Badge>;
  }
  return <Badge variant="outline" className="font-normal">{state.toLowerCase()}</Badge>;
}

export default function Deploy() {
  const [ucCatalog, setUcCatalog] = useState<string>("<catalog>");
  const [ucSchema, setUcSchema] = useState<string>("<schema>");
  useEffect(() => {
    api.health().then((h) => { setUcCatalog(h.uc_catalog); setUcSchema(h.uc_schema); }).catch(() => {});
  }, []);
  const [models, setModels] = useState<DeployableModel[]>([]);
  const [endpoints, setEndpoints] = useState<ServingEndpointRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // Deploy dialog
  const [open, setOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<DeployableModel | null>(null);
  const [selectedVersion, setSelectedVersion] = useState<string>("");
  const [endpointName, setEndpointName] = useState<string>("");
  const [workloadType, setWorkloadType] = useState<string>("GPU_LARGE");
  const [workloadSize, setWorkloadSize] = useState<string>("Small");
  const [scaleToZero, setScaleToZero] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState(false);

  const refresh = () => {
    setLoading(true);
    Promise.all([
      api.deployableModels().then(setModels).catch((e) => setError(`models: ${e.message}`)),
      api.servingEndpoints().then(setEndpoints).catch((e) => setError(`endpoints: ${e.message}`)),
    ]).finally(() => setLoading(false));
  };
  useEffect(refresh, []);

  // Auto-refresh while any endpoint is updating
  useEffect(() => {
    const updating = endpoints.some(
      (e) => e.config_state === "IN_PROGRESS" || e.state === "NOT_READY"
    );
    if (!updating) return;
    const t = setInterval(() => {
      api.servingEndpoints().then(setEndpoints).catch(() => {});
    }, 15000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoints.map((e) => `${e.name}:${e.state}:${e.config_state}`).join(",")]);

  const openDeploy = (m: DeployableModel) => {
    setSelectedModel(m);
    setSelectedVersion(m.versions[0] || "");
    setEndpointName("");
    setError(null);
    setInfo(null);
    setOpen(true);
  };

  const submit = async () => {
    if (!selectedModel) return;
    setSubmitting(true);
    setError(null);
    try {
      const r = await api.deploySubmit({
        model_full_name: selectedModel.full_name,
        model_version: selectedVersion || undefined,
        endpoint_name: endpointName.trim() || undefined,
        workload_type: workloadType,
        workload_size: workloadSize,
        scale_to_zero: scaleToZero,
      });
      setInfo(`Deploying ${r.name} → ${r.model_full_name} v${r.version}. Endpoint will be ready in ~5–15 min; it shows up in Playground's model list once state=READY.`);
      setOpen(false);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const deleteEndpoint = async (name: string) => {
    if (!confirm(`Tear down endpoint '${name}'? This stops serving immediately.`)) return;
    try {
      await api.deployDelete(name);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Deploy</h1>
        <p className="text-sm text-muted-foreground">
          Spin up a Databricks Model Serving endpoint for any UC-registered model — base Qwen3-VL
          (after running the register-base-model notebook) or a fine-tuned variant from the Train
          tab. Once deployed, the endpoint appears in Playground's AI Gateway dropdown so you can
          A/B test against base + AI Gateway models directly.
        </p>
      </div>

      {info && (
        <Card>
          <CardContent className="flex items-start gap-2 p-3 text-sm">
            <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-500" />
            <span>{info}</span>
          </CardContent>
        </Card>
      )}
      {error && (
        <Card>
          <CardContent className="flex items-start gap-2 p-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <span>{error}</span>
          </CardContent>
        </Card>
      )}

      {/* Models registered in UC */}
      <Card>
        <CardHeader className="flex flex-row items-start justify-between pb-3">
          <div>
            <CardTitle className="text-base">Registered models</CardTitle>
            <CardDescription>
              Models in <span className="font-mono">{ucCatalog}.{ucSchema}</span> (fine-tunes
              from the Train tab + anything else you've registered). Click Deploy to spin up a
              serving endpoint.
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCcw className="size-3.5" />}
            Refresh
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          {models.length === 0 ? (
            <p className="p-4 text-xs text-muted-foreground">
              {loading ? "loading…" : "No UC models found. Finish a Train run or register a base model."}
            </p>
          ) : (
            <div className="divide-y">
              {models.map((m) => (
                <div key={m.full_name} className="flex flex-wrap items-center gap-3 px-4 py-3">
                  <Cloud className="size-4 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="truncate font-mono text-sm">{m.name}</span>
                      {m.versions.length > 0 && (
                        <Badge variant="outline" className="font-normal">v{m.versions[0]}</Badge>
                      )}
                      {m.base_model && (
                        <span className="text-xs text-muted-foreground">from {m.base_model}</span>
                      )}
                      {m.n_train && (
                        <span className="text-xs text-muted-foreground">· {m.n_train} train</span>
                      )}
                      {m.train_loss != null && (
                        <span className="text-xs text-muted-foreground">· loss {m.train_loss.toFixed(3)}</span>
                      )}
                    </div>
                    <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                      {m.full_name}
                      {m.updated_at && ` · updated ${new Date(m.updated_at).toLocaleString()}`}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    onClick={() => openDeploy(m)}
                    disabled={m.versions.length === 0}
                    className="gap-1"
                  >
                    <Rocket className="size-3.5" /> Deploy
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Live endpoints */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Serving endpoints</CardTitle>
          <CardDescription>
            All workspace endpoints. Workbench-managed endpoints show a ⚡ badge and can be deleted
            from here; others have to be managed in the Databricks UI.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          {endpoints.length === 0 ? (
            <p className="p-4 text-xs text-muted-foreground">
              {loading ? "loading…" : "No serving endpoints in this workspace yet."}
            </p>
          ) : (
            <div className="divide-y">
              {endpoints.map((e) => (
                <div key={e.name} className="flex flex-wrap items-center gap-3 px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="truncate font-mono text-sm">{e.name}</span>
                      <StateBadge state={e.state} config_state={e.config_state} />
                      {e.managed && (
                        <Badge variant="secondary" className="font-normal text-[10px]" title="Managed by this app">⚡ managed</Badge>
                      )}
                      {e.workload_type && (
                        <span className="text-xs text-muted-foreground">{e.workload_type}</span>
                      )}
                      {e.workload_size && (
                        <span className="text-xs text-muted-foreground">· {e.workload_size}</span>
                      )}
                    </div>
                    <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                      {e.model && <>{e.model}{e.version && ` v${e.version}`}</>}
                      {e.creator && ` · creator: ${e.creator}`}
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {e.invocation_url && (
                      <a
                        href={e.invocation_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
                      >
                        <ExternalLink className="size-3" /> URL
                      </a>
                    )}
                    {e.managed && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => deleteEndpoint(e.name)}
                        className="gap-1 text-muted-foreground"
                        title="Tear down this endpoint"
                      >
                        <Trash2 className="size-3.5" />
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex items-start gap-2 p-3 text-xs text-muted-foreground">
          <Wand2 className="mt-0.5 size-3.5 shrink-0" />
          <p>
            <span className="font-medium">A/B in Playground:</span> once an endpoint is{" "}
            <code>READY</code>, it shows up in Playground's AI Gateway model dropdown. Select it
            alongside the base model + your fine-tuned variant and run on the labeled frame set —
            the accuracy footer + ✓/✗ chips show the lift directly.
          </p>
        </CardContent>
      </Card>

      {/* Deploy dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Deploy model</DialogTitle>
            <DialogDescription>
              Provisions a Databricks Model Serving endpoint. Cold-start typically 5–15 min for
              VLMs; the endpoint will scale to zero between invocations to keep cost low.
            </DialogDescription>
          </DialogHeader>
          {selectedModel && (
            <div className="space-y-3">
              <div>
                <Label className="text-xs">Model</Label>
                <p className="rounded border bg-muted/40 p-2 font-mono text-xs">
                  {selectedModel.full_name}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs">Version</Label>
                  <Select value={selectedVersion} onValueChange={setSelectedVersion}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {selectedModel.versions.map((v) => (
                        <SelectItem key={v} value={v}>v{v}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-xs">Endpoint name (optional)</Label>
                  <Input
                    placeholder={`auto (${selectedModel.name.toLowerCase().slice(0, 30)}-…)`}
                    value={endpointName}
                    onChange={(e) => setEndpointName(e.target.value)}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs">Workload type</Label>
                  <Select value={workloadType} onValueChange={setWorkloadType}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {WORKLOAD_TYPES.map((t) => (
                        <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-xs">Workload size</Label>
                  <Select value={workloadSize} onValueChange={setWorkloadSize}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {WORKLOAD_SIZES.map((s) => (
                        <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <label className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={scaleToZero}
                  onChange={(e) => setScaleToZero(e.target.checked)}
                />
                Scale to zero when idle (cheaper; first invocation triggers a cold start)
              </label>
              {error && (
                <p className="flex items-center gap-1 text-xs text-destructive">
                  <AlertCircle className="size-3.5" /> {error}
                </p>
              )}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)} disabled={submitting}>
              Cancel
            </Button>
            <Button onClick={submit} disabled={submitting || !selectedModel} className="gap-2">
              {submitting ? <><Loader2 className="size-4 animate-spin" /> Deploying…</> : <><Rocket className="size-4" /> Deploy</>}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
