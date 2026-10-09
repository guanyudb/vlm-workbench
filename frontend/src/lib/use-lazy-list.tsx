import { useEffect, useRef, useState } from "react";

/**
 * useLazyList: lazy-load items with IntersectionObserver
 * Returns the items that should be rendered given the current scroll position
 */
export function useLazyList<T>(
  items: T[],
  pageSize: number = 50
) {
  const [displayCount, setDisplayCount] = useState(pageSize);
  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && displayCount < items.length) {
          setDisplayCount((prev) => Math.min(prev + pageSize, items.length));
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [displayCount, items.length, pageSize]);

  const displayedItems = items.slice(0, displayCount);
  const hasMore = displayCount < items.length;

  return { displayedItems, sentinelRef, hasMore };
}
