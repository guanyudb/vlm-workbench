import { useEffect, useState } from "react";
import {
  CheckCircle2, ChevronDown, ChevronRight, ExternalLink, Info, Loader2,
  RefreshCcw, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import {
  Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { api, type TaskConfig } from "@/api";
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
              // Single inline action lives on the HF row — opening the dialog
              // there also kicks off the model cache job, so duplicating the
              // button on the Local model cache row was pure noise.
              const hasInlineAction = /HuggingFace/i.test(c.name);
              return (
                <div key={c.name} className="px-4 py-3">
                  <div className="flex w-full items-start gap-3">
                    <button
                      onClick={() => c.remediation && toggle(c.name)}
                      className={cn(
                        "flex flex-1 items-start gap-3 text-left",
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
                    {/* Always-available action button — works whether the check is green or red */}
                    {hasInlineAction && <HFTokenDialog onSaved={refresh} compact />}
                  </div>
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

      <TaskConfigCard />

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

// ── HF token paste dialog ────────────────────────────────────────────────
//
// Posts the token to /api/setup/hf-token. The endpoint creates the scope if
// it doesn't exist (App SP becomes the owner), writes the token, and — if
// `models` is non-empty — kicks off a setup_cache job to download those
// weights into the Volume. Token + model selection are paired in one
// dialog because HF tokens are needed to unlock gated repos like MedGemma.
function HFTokenDialog({ onSaved, compact }: { onSaved: () => void; compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const [token, setToken] = useState("");
  const [presets, setPresets] = useState<{ name: string; hf_repo: string; label: string }[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [okMsg, setOkMsg] = useState<string | null>(null);
  const [runUrl, setRunUrl] = useState<string | null>(null);

  useEffect(() => {
    if (open && presets.length === 0) {
      api.getLocalModelPresets()
        .then((r) => {
          setPresets(r.presets);
          setSelected(new Set(r.presets.map((p) => p.name)));  // default: cache all
        })
        .catch(() => { /* presets are optional */ });
    }
  }, [open]);

  const toggle = (name: string) => {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const save = async () => {
    setSaving(true);
    setErr(null);
    setOkMsg(null);
    setRunUrl(null);
    try {
      const models = presets
        .filter((p) => selected.has(p.name))
        // Spread the preset so optional fields (accelerator,
        // base_environment, inference_notebook) flow through to the backend
        // and get baked into the manifest.yaml that setup_cache writes.
        // Without this, Gemma 4 12B would default to GPU_1xA10 and OOM.
        .map((p) => ({ ...p }));
      const r = await api.storeHFToken(token.trim(), models.length ? models : undefined);
      const partials: string[] = [`token stored in ${r.scope}/${r.key} (${r.len} chars)`];
      if (r.models_run_id) {
        partials.push(`caching ${models.length} model(s) — run ${r.models_run_id}`);
        if (r.models_run_url) setRunUrl(r.models_run_url);
      }
      if (r.models_error) partials.push(`models error: ${r.models_error}`);
      setOkMsg(partials.join(" · "));
      setToken("");
      onSaved();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {compact ? (
        <Button variant="outline" size="sm" className="shrink-0" onClick={() => setOpen(true)}>
          Set token / cache
        </Button>
      ) : (
        <Button size="sm" className="mt-3" onClick={() => setOpen(true)}>
          Set token in-app
        </Button>
      )}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>HuggingFace token + model cache</DialogTitle>
            <DialogDescription>
              Pasted into the workspace secret scope this App owns. We never log it. Get one from{" "}
              <a className="text-primary underline" href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">huggingface.co/settings/tokens</a>.
              Models you check below will start downloading to the Volume immediately so they're ready in the Playground's Local section.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <label className="text-xs font-medium">HF token</label>
              <Input
                type="password"
                placeholder="hf_..."
                value={token}
                onChange={(e) => setToken(e.target.value)}
                spellCheck={false}
                autoFocus
                className="mt-1"
              />
            </div>
            {presets.length > 0 && (
              <div>
                <label className="text-xs font-medium">Cache these models now (optional)</label>
                <div className="mt-2 space-y-1.5 rounded-md border bg-muted/30 px-3 py-2">
                  {presets.map((p) => (
                    <label key={p.name} className="flex items-center gap-2 text-xs">
                      <input
                        type="checkbox"
                        checked={selected.has(p.name)}
                        onChange={() => toggle(p.name)}
                        className="size-3.5"
                      />
                      <span className="font-medium">{p.label}</span>
                      <code className="text-[10px] text-muted-foreground">{p.hf_repo}</code>
                    </label>
                  ))}
                </div>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Each model is multi-GB and takes 5–30min to snapshot. The job runs in the background; close this dialog after click and watch progress in the Workflow run.
                </p>
              </div>
            )}
            {err && <p className="text-xs text-destructive">{err}</p>}
            {okMsg && (
              <p className="text-xs text-emerald-600">
                {okMsg}
                {runUrl && (
                  <>
                    {" · "}
                    <a className="underline" href={runUrl} target="_blank" rel="noopener noreferrer">open run</a>
                  </>
                )}
              </p>
            )}
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" size="sm">Close</Button>
            </DialogClose>
            <Button size="sm" disabled={!token.trim() || saving} onClick={save}>
              {saving ? <Loader2 className="size-3 animate-spin" /> : null}
              {selected.size > 0 ? `Save + cache ${selected.size}` : "Save token"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
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

// ── Task definition card ─────────────────────────────────────────────────
//
// Edits the workspace's task_config.json on the UC Volume. Same source of
// truth that Playground's prompt + Label's vocab picker + the optimizer's
// gold-label vocabulary all read from. Saving writes to
// `<volume>/config/task_config.json` and busts the backend cache.
function TaskConfigCard() {
  const [cfg, setCfg] = useState<TaskConfig | null>(null);
  const [vocabText, setVocabText] = useState("");
  const [promptText, setPromptText] = useState("");
  const [schemaText, setSchemaText] = useState("");  // JSON blob editor
  const [schemaErr, setSchemaErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.getTaskConfig()
      .then((c) => {
        setCfg(c);
        setVocabText(c.vocabulary.join("\n"));
        setPromptText(c.prompt_template);
        setSchemaText(JSON.stringify(c.response_schema, null, 2));
      })
      .catch((e) => setErr((e as Error).message))
      .finally(() => setLoading(false));
  }, []);

  // Live-validate the schema JSON so the user knows before Save fails
  useEffect(() => {
    if (!schemaText.trim()) { setSchemaErr(null); return; }
    try {
      const parsed = JSON.parse(schemaText);
      if (!parsed || typeof parsed !== "object") throw new Error("must be an object");
      if (!parsed.primary_class_path) throw new Error("primary_class_path is required");
      setSchemaErr(null);
    } catch (e) {
      setSchemaErr((e as Error).message);
    }
  }, [schemaText]);

  const save = async () => {
    setSaving(true);
    setSaveMsg(null);
    setErr(null);
    try {
      const vocabulary = vocabText.split("\n").map((s) => s.trim()).filter(Boolean);
      const response_schema = JSON.parse(schemaText);
      const c = await api.saveTaskConfig({ vocabulary, prompt_template: promptText, response_schema });
      setCfg(c);
      setSaveMsg(`saved · ${vocabulary.length} vocab entries`);
      setTimeout(() => setSaveMsg(null), 4000);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const reset = () => {
    if (!cfg) return;
    setVocabText(cfg.vocabulary.join("\n"));
    setPromptText(cfg.prompt_template);
    setSchemaText(JSON.stringify(cfg.response_schema, null, 2));
    setSaveMsg(null);
    setErr(null);
  };

  const dirty = cfg && (
    vocabText !== cfg.vocabulary.join("\n") ||
    promptText !== cfg.prompt_template ||
    schemaText !== JSON.stringify(cfg.response_schema, null, 2)
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Task definition</CardTitle>
            <CardDescription>
              Vocabulary + prompt template the entire workbench reads from. Edits propagate to Playground, Label, and the optimizer without a redeploy.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {saveMsg && <span className="text-xs text-emerald-600">{saveMsg}</span>}
            {dirty && <Badge variant="secondary" className="font-normal">unsaved</Badge>}
            <Button variant="ghost" size="sm" onClick={reset} disabled={!dirty || saving}>
              Reset
            </Button>
            <Button size="sm" onClick={save} disabled={!dirty || saving || loading || !!schemaErr}>
              {saving ? <Loader2 className="size-3 animate-spin" /> : null} Save
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {loading && <p className="text-xs text-muted-foreground">Loading…</p>}
        {err && (
          <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
            {err}
          </div>
        )}
        {!loading && (
          <>
            <div>
              <label className="text-xs font-medium">Vocabulary (one class per line)</label>
              <p className="mt-0.5 text-xs text-muted-foreground">
                First 9 entries become the digit-key hotkeys in the Label tab.
              </p>
              <Textarea
                value={vocabText}
                onChange={(e) => setVocabText(e.target.value)}
                rows={6}
                className="mt-2 font-mono text-xs"
                spellCheck={false}
              />
            </div>
            <div>
              <label className="text-xs font-medium">Response schema</label>
              <p className="mt-0.5 text-xs text-muted-foreground">
                <code className="rounded bg-muted px-1 py-0.5">shape</code> is sent to the LLM (substituted into <code className="rounded bg-muted px-1 py-0.5">{"{{response_schema}}"}</code> in the prompt). The <code className="rounded bg-muted px-1 py-0.5">*_path</code> lenses tell Playground / Label / Studio how to extract fields from the parsed response.
              </p>
              <Textarea
                value={schemaText}
                onChange={(e) => setSchemaText(e.target.value)}
                rows={14}
                className={cn("mt-2 font-mono text-xs", schemaErr && "border-destructive")}
                spellCheck={false}
              />
              {schemaErr && (
                <p className="mt-1 text-xs text-destructive">JSON: {schemaErr}</p>
              )}
            </div>
            <div>
              <label className="text-xs font-medium">Prompt template</label>
              <p className="mt-0.5 text-xs text-muted-foreground">
                <code className="rounded bg-muted px-1 py-0.5">{"{{vocabulary}}"}</code> and <code className="rounded bg-muted px-1 py-0.5">{"{{response_schema}}"}</code> are substituted at request time. This is what Playground's “Reset to default” restores.
              </p>
              <Textarea
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                rows={14}
                className="mt-2 font-mono text-xs"
                spellCheck={false}
              />
            </div>
            {cfg?.rendered_prompt && (
              <details className="rounded-md border bg-muted/40 px-3 py-2 text-xs">
                <summary className="cursor-pointer font-medium">Rendered preview</summary>
                <pre className="mt-2 whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-muted-foreground">{cfg.rendered_prompt}</pre>
              </details>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
