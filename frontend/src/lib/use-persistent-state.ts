import { useEffect, useState } from "react";

/** Like useState, but writes the latest value to localStorage under `key`
 * and rehydrates from it on mount. JSON-serialized; pass a custom
 * serializer/deserializer if your value isn't a plain JSON value (e.g. Set,
 * Map). Failures (quota, parse errors) are swallowed — falls back to the
 * default value. */
export function usePersistentState<T>(
  key: string,
  defaultValue: T,
  opts?: {
    serialize?: (v: T) => string;
    deserialize?: (s: string) => T;
  },
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const serialize = opts?.serialize ?? JSON.stringify;
  const deserialize = opts?.deserialize ?? JSON.parse;
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw == null) return defaultValue;
      return deserialize(raw) as T;
    } catch {
      return defaultValue;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, serialize(value));
    } catch { /* quota — silent */ }
  }, [key, value, serialize]);
  return [value, setValue];
}

/** JSON serialize/deserialize helpers for Set<string>. */
export const setStringCodec = {
  serialize: (s: Set<string>) => JSON.stringify(Array.from(s)),
  deserialize: (raw: string): Set<string> => new Set<string>(JSON.parse(raw)),
};

/** JSON serialize/deserialize for Map<string, T>. T must be JSON-safe. */
export function mapStringCodec<T>() {
  return {
    serialize: (m: Map<string, T>) => JSON.stringify(Array.from(m.entries())),
    deserialize: (raw: string): Map<string, T> => new Map<string, T>(JSON.parse(raw)),
  };
}
