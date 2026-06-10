import { useEffect, useState } from "react";
import { AlertTriangle, ChevronRight, X } from "lucide-react";
import { api } from "@/api";

const DISMISS_KEY = "vlmwb.setup-banner.dismissed";

// Polls /api/setup/check every 60s. When any check fails, renders a top
// banner pointing the user to the Setup tab. Idea borrowed from
// genesis-workbench's "Profile setup is incomplete" pattern — non-blocking
// but persistent. Auto-hides on routes where the user is already at Setup.
//
// We intentionally do NOT disable destructive actions on other tabs — the
// user might want to click into Library to see a partial state, or open
// Playground to read past results. The banner just nudges.
export function SetupGateBanner({
  currentRoute,
  goToSetup,
}: {
  currentRoute: string;
  goToSetup: () => void;
}) {
  const [failing, setFailing] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  // Session-scoped dismiss: hide the banner for the rest of the tab session
  // (not localStorage — we want it to reappear on next visit if checks are
  // still failing, since the user might have forgotten about the issue).
  const [dismissed, setDismissed] = useState(() => {
    try { return sessionStorage.getItem(DISMISS_KEY) === "1"; } catch { return false; }
  });

  useEffect(() => {
    let alive = true;
    const fetchOnce = () => {
      api.setupCheck()
        .then((r) => {
          if (!alive) return;
          setFailing(r.checks.filter((c) => !c.ok).map((c) => c.name));
        })
        .catch(() => { /* leave previous state */ })
        .finally(() => { if (alive) setLoading(false); });
    };
    fetchOnce();
    const t = setInterval(fetchOnce, 60000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  if (loading || failing.length === 0 || currentRoute === "setup" || dismissed) return null;

  const dismiss = () => {
    try { sessionStorage.setItem(DISMISS_KEY, "1"); } catch { /* private mode */ }
    setDismissed(true);
  };

  // Cap displayed names so the banner stays one line. The Setup tab shows
  // the full list with remediation copy.
  const shown = failing.slice(0, 3);
  const rest = failing.length - shown.length;

  return (
    <div className="border-b border-amber-500/40 bg-amber-50 px-4 py-2 dark:bg-amber-950/30">
      <div className="flex items-start gap-2 text-xs">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
        <div className="flex-1">
          <span className="font-medium text-amber-900 dark:text-amber-100">
            Setup is incomplete.
          </span>{" "}
          <span className="text-amber-800 dark:text-amber-200">
            {failing.length} check{failing.length === 1 ? "" : "s"} failing
            {": "}
            {shown.join(", ")}
            {rest > 0 && `, +${rest} more`}.
          </span>{" "}
          <span className="text-amber-700 dark:text-amber-300">
            Some features (upload, train, deploy) may not work until you fix these.
          </span>
        </div>
        <button
          onClick={goToSetup}
          className="inline-flex shrink-0 items-center gap-1 rounded-md border border-amber-700/40 bg-amber-100/60 px-2 py-1 text-amber-900 hover:bg-amber-100 dark:border-amber-500/30 dark:bg-amber-900/20 dark:text-amber-100 dark:hover:bg-amber-900/40"
        >
          Open Setup <ChevronRight className="size-3" />
        </button>
        <button
          onClick={dismiss}
          className="inline-flex shrink-0 items-center rounded-md p-1 text-amber-700 hover:bg-amber-200/40 dark:text-amber-300 dark:hover:bg-amber-900/40"
          title="Dismiss for this session — banner reappears on next visit if any checks still fail"
        >
          <X className="size-3.5" />
        </button>
      </div>
    </div>
  );
}
