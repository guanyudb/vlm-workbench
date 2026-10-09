/**
 * Shared instrument parsing utilities across Compare, Playground, and Studio tabs.
 * Extracts structured response fields from LLM model responses.
 */

export function extractInstrument(parsed: any): string {
  if (!parsed || typeof parsed !== "object") return "?";
  if (typeof parsed.instrument === "string") {
    return parsed.instrument.trim() || "no_instrument_visible";
  }
  const list = parsed.instruments;
  if (Array.isArray(list)) {
    if (list.length === 0) return "no_instrument_visible";
    if (typeof list[0] === "object") {
      const c = list[0]?.class;
      if (typeof c === "string") return c.trim() || "no_instrument_visible";
    }
  }
  return "no_instrument_visible";
}

export function extractEvidence(parsed: any): string {
  if (!parsed) return "";
  if (typeof parsed.evidence === "string") return parsed.evidence;
  const list = parsed.instruments;
  if (Array.isArray(list) && list.length && typeof list[0] === "object") {
    return list[0]?.evidence ?? "";
  }
  return "";
}

export function extractAnatomy(parsed: any): string {
  if (!parsed) return "";
  return parsed.anatomy || "";
}
