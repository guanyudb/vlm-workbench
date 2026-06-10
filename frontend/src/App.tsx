import { useEffect, useState } from "react";
import { Beaker, BarChart3, Settings, Wrench, MessageSquare, Film, Tag, Sparkles, Video, Rocket } from "lucide-react";
import { ThemeProvider } from "@/components/apx/theme-provider";
import { Navbar } from "@/components/apx/navbar";
import { OptimizeStatusIndicator } from "@/components/apx/optimize-status-indicator";
import { RunsIndicator } from "@/components/apx/runs-indicator";
import { SetupGateBanner } from "@/components/apx/setup-gate-banner";
import { OptimizeTrackerProvider } from "@/lib/optimize-tracker";
import { RunsTrackerProvider } from "@/lib/runs-tracker";
import { api } from "@/api";
import Playground from "@/Playground";
import Studio from "@/Studio";
import Compare from "@/Compare";
import LabelTab from "@/Label";
import Train from "@/Train";
import VideosTab from "@/Videos";
import Deploy from "@/Deploy";
import Setup from "@/Setup";
import { cn } from "@/lib/utils";

const PHASE2 = ["library", "compare", "label", "train", "deploy", "videos", "setup", "eval", "jobs", "agent", "settings"] as const;
type Route = "playground" | "studio" | (typeof PHASE2)[number];

// Two-tier route layout: the active tabs make up the primary nav block;
// the `comingSoon` block lives in a small footer at the bottom of the
// sidebar so unfinished features don't compete visually with the working
// ones.
const ROUTES: { id: Route; label: string; icon: any }[] = [
  { id: "videos", label: "Library", icon: Video },
  { id: "playground", label: "Playground", icon: Beaker },
  { id: "label", label: "Label", icon: Tag },
  // "Train" → "Refine". Avoids ML jargon while still implying "improve
  // the model"; flows with the workflow narrative. Route id stays `train`
  // so saved localStorage routes + bookmarks don't break.
  { id: "train", label: "Refine", icon: Sparkles },
  { id: "deploy", label: "Deploy", icon: Rocket },
  { id: "studio", label: "Studio", icon: Film },
  { id: "compare", label: "Compare", icon: BarChart3 },
  { id: "setup", label: "Setup", icon: Settings },
];

const COMING_SOON: { id: Route; label: string; icon: any }[] = [
  { id: "eval", label: "Eval", icon: Wrench },
  { id: "agent", label: "Agent", icon: MessageSquare },
];

export default function App() {
  return (
    <ThemeProvider defaultTheme="light">
      <RunsTrackerProvider>
        <OptimizeTrackerProvider>
          <Main />
        </OptimizeTrackerProvider>
      </RunsTrackerProvider>
    </ThemeProvider>
  );
}

const ROUTE_KEY = "vlmwb.route.v1";

function Main() {
  const [route, setRoute] = useState<Route>(() => {
    const saved = localStorage.getItem(ROUTE_KEY);
    const valid = ["playground", "studio", "compare", "label", "train", "videos", "deploy", "setup"];
    if (saved && valid.includes(saved)) return saved as Route;
    return "playground";
  });
  // Track which routes have been visited at least once. We mount lazily on
  // first visit, then keep the component mounted (just hidden) so its state
  // — selections, results, prompt drafts — survives tab switches.
  const [mounted, setMounted] = useState<Set<Route>>(() => new Set([route]));
  const [healthy, setHealthy] = useState<boolean | null>(null);
  // Cross-route state: when Playground sends a snapshot to Studio, it lands here.
  const [pendingSnapshotId, setPendingSnapshotId] = useState<string | null>(null);
  // When the user clicks "Apply" on a finished optimize job from anywhere in
  // the app, we route them to Playground with the prompt staged.
  const [pendingOptimizedPrompt, setPendingOptimizedPrompt] = useState<string | null>(null);

  const navigate = (next: Route) => {
    setMounted((cur) => (cur.has(next) ? cur : new Set([...cur, next])));
    setRoute(next);
    try { localStorage.setItem(ROUTE_KEY, next); } catch { /* quota */ }
  };

  const goToStudioWithSnapshot = (snapshotId: string) => {
    setPendingSnapshotId(snapshotId);
    navigate("studio");
  };

  const applyOptimizedPrompt = (p: string) => {
    setPendingOptimizedPrompt(p);
    navigate("playground");
  };

  useEffect(() => {
    api.health().then(() => setHealthy(true)).catch(() => setHealthy(false));
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Navbar
        title="Surgical Vision Workbench"
        subtitle={
          <span className="flex items-center gap-1.5">
            From raw surgical video to a fine-tuned VLM, in one workspace
            <span
              className={cn(
                "inline-block size-1.5 rounded-full",
                healthy === true && "bg-emerald-500",
                healthy === false && "bg-red-500",
                healthy === null && "bg-amber-500 animate-pulse"
              )}
              title={healthy === true ? "backend healthy" : healthy === false ? "backend unreachable" : "checking…"}
            />
          </span>
        }
        right={
          <div className="flex items-center gap-2">
            <RunsIndicator />
            <OptimizeStatusIndicator onApplyPrompt={applyOptimizedPrompt} />
          </div>
        }
      />

      <SetupGateBanner currentRoute={route} goToSetup={() => navigate("setup")} />

      <div className="flex">
        <aside className="sticky top-14 hidden h-[calc(100vh-3.5rem)] w-48 shrink-0 flex-col border-r bg-card/40 lg:flex">
          <nav className="flex flex-1 flex-col gap-0.5 p-2">
            {ROUTES.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => navigate(id)}
                className={cn(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors hover:bg-accent hover:text-accent-foreground",
                  route === id ? "bg-accent text-accent-foreground" : "text-muted-foreground",
                )}
              >
                <Icon className="size-4" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
          {/* Coming-soon block — placeholder for not-yet-shipped tabs, kept
              in a separate footer so it doesn't visually compete with the
              active nav above. */}
          <div className="border-t p-2">
            <p className="px-2.5 pb-1 text-[10px] uppercase tracking-wider text-muted-foreground/70">
              Coming soon
            </p>
            <div className="flex flex-col gap-0.5">
              {COMING_SOON.map(({ id, label, icon: Icon }) => (
                <div
                  key={id}
                  className="flex cursor-not-allowed items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-muted-foreground/60"
                  title="Coming in a later phase"
                >
                  <Icon className="size-4" />
                  <span>{label}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <main className="min-w-0 flex-1 px-4 py-6 sm:px-6">
          {/* Each visited tab stays mounted; we hide inactive ones via CSS so
              their internal state (selections, run results, scroll position)
              survives navigation. */}
          {mounted.has("playground") && (
            <div className={cn(route === "playground" ? "block" : "hidden")}>
              <Playground
                onSendToStudio={goToStudioWithSnapshot}
                pendingPrompt={pendingOptimizedPrompt}
                onConsumePendingPrompt={() => setPendingOptimizedPrompt(null)}
              />
            </div>
          )}
          {mounted.has("studio") && (
            <div className={cn(route === "studio" ? "block" : "hidden")}>
              <Studio
                initialSnapshotId={pendingSnapshotId}
                onConsumeInitialSnapshot={() => setPendingSnapshotId(null)}
              />
            </div>
          )}
          {mounted.has("compare") && (
            <div className={cn(route === "compare" ? "block" : "hidden")}>
              <Compare />
            </div>
          )}
          {mounted.has("label") && (
            <div className={cn(route === "label" ? "block" : "hidden")}>
              <LabelTab />
            </div>
          )}
          {mounted.has("train") && (
            <div className={cn(route === "train" ? "block" : "hidden")}>
              <Train />
            </div>
          )}
          {mounted.has("videos") && (
            <div className={cn(route === "videos" ? "block" : "hidden")}>
              <VideosTab />
            </div>
          )}
          {mounted.has("deploy") && (
            <div className={cn(route === "deploy" ? "block" : "hidden")}>
              <Deploy />
            </div>
          )}
          {mounted.has("setup") && (
            <div className={cn(route === "setup" ? "block" : "hidden")}>
              <Setup />
            </div>
          )}
          {route !== "playground" && route !== "studio" && route !== "compare" && route !== "label" && route !== "train" && route !== "videos" && route !== "deploy" && route !== "setup" && <Placeholder route={route} />}
        </main>
      </div>
    </div>
  );
}

function Placeholder({ route }: { route: Route }) {
  return (
    <div className="rounded-lg border border-dashed p-12 text-center text-muted-foreground">
      <p className="text-sm">"{route}" lands in a later phase.</p>
    </div>
  );
}
