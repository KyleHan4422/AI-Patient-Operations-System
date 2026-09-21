export type ProbeResult = {
  ok: boolean;
  latency_ms: number | null;
  error: string | null;
};

export type HealthReport = {
  status: "ok" | "degraded" | "unhealthy";
  version: string;
  checks: Record<string, ProbeResult>;
  degraded: string[];
};

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

/**
 * A 503 from /health is a valid, parseable answer ("I am unhealthy"), not a
 * transport failure -- so we read the body on any status that returned JSON.
 * Only an actual network error means we learned nothing.
 */
export async function fetchHealth(): Promise<HealthReport> {
  const res = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
  return (await res.json()) as HealthReport;
}
