import { useEffect, useRef } from "react";

/**
 * Poll `fn` every `interval` ms with visibility awareness and error backoff.
 *
 * - While the tab is hidden, ticks are skipped (no requests); becoming visible
 *   polls immediately.
 * - While `stopWhen()` is true, ticks are skipped but the loop keeps running,
 *   so polling resumes by itself once the condition flips (e.g. a job is
 *   submitted after the page went idle).
 * - `fn` throwing backs off ×2 per consecutive failure, capped at 60 s.
 */
export function usePolling(
  fn: () => Promise<void>,
  interval: number,
  stopWhen?: () => boolean
) {
  const fnRef = useRef(fn);
  const stopWhenRef = useRef(stopWhen);
  fnRef.current = fn;
  stopWhenRef.current = stopWhen;

  useEffect(() => {
    let cancelled = false;
    let running = false;
    let timer: number | undefined;
    let errors = 0;

    const schedule = (ms: number) => {
      if (!cancelled) timer = window.setTimeout(tick, ms);
    };

    const tick = async () => {
      if (cancelled || running) return;
      if (document.hidden || stopWhenRef.current?.()) {
        schedule(interval);
        return;
      }
      running = true;
      try {
        await fnRef.current();
        errors = 0;
        schedule(interval);
      } catch {
        errors += 1;
        schedule(Math.min(interval * 2 ** (errors - 1), 60000));
      } finally {
        running = false;
      }
    };

    const onVisibility = () => {
      if (!document.hidden && !running) {
        window.clearTimeout(timer);
        tick();
      }
    };

    document.addEventListener("visibilitychange", onVisibility);
    tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [interval]);
}
