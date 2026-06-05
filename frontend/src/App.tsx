import { useEffect, useState } from "react";
import { Beaker, BookOpen, BarChart3, Settings, Wrench, Briefcase, MessageSquare, Film, Tag, Sparkles, Video, Rocket } from "lucide-react";
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

// "Jobs" used to live here as a coming-soon item — the navbar runs pill
// already surfaces in-flight + recent job runs across all tabs, so a
// dedicated tab would be redundant. Eval, the second "Library" (docs/
// playbooks), and Agent stay as soon-markers because they're distinct
// future features, not duplicates of something already shipped.
const ROUTES: { id: Route; label: string; icon: any; available: boolean }[] = [
  { id: "videos", label: "Library", icon: Video, available: true },
  { id: "playground", label: "Playground", icon: Beaker, available: true },
  { id: "label", label: "Label", icon: Tag, available: true },
  { id: "train", label: "Train", icon: Sparkles, available: true },
  { id: "deploy", label: "Deploy", icon: Rocket, available: true },
  { id: "studio", label: "Studio", icon: Film, available: true },
  { id: "compare", label: "Compare", icon: BarChart3, available: true },
  { id: "setup", label: "Setup", icon: Settings, available: true },
  // The Videos tab (top of the list) IS our Library — the second "library"
  // greyed-out entry was originally a placeholder for a docs/playbooks
  // page. Removed to stop confusing users into thinking Library was
  // unavailable.
  { id: "eval", label: "Eval", icon: Wrench, available: false },
  { id: "agent", label: "Agent", icon: MessageSquare, available: false },
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
        title="Surgical VLM Workbench"
        subtitle={
          <span className="flex items-center gap-1.5">
            Phase 1: Playground + Studio
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
        <aside className="sticky top-14 hidden h-[calc(100vh-3.5rem)] w-48 shrink-0 border-r bg-card/40 lg:block">
          <nav className="flex flex-col gap-0.5 p-2">
            {ROUTES.map(({ id, label, icon: Icon, available }) => (
              <button
                key={id}
                onClick={() => available && navigate(id)}
                disabled={!available}
                className={cn(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors",
                  route === id ? "bg-accent text-accent-foreground" : "text-muted-foreground",
                  available ? "hover:bg-accent hover:text-accent-foreground" : "opacity-50 cursor-not-allowed"
                )}
                title={!available ? "Coming in a later phase" : undefined}
              >
                <Icon className="size-4" />
                <span>{label}</span>
                {!available && <span className="ml-auto text-[10px] text-muted-foreground">soon</span>}
              </button>
            ))}
          </nav>
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
