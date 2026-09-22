"use client";

import { useEffect, useState } from "react";
import { fetchHealth, type HealthReport } from "@/lib/health";

/**
 * Polls /health. `unreachable` means the API itself did not answer -- a
 * different condition from `unhealthy`, which the API reports about itself.
 */
export function useHealth(pollMs = 10_000) {
  const [report, setReport] = useState<HealthReport | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await fetchHealth();
        if (cancelled) return;
        setReport(next);
        setUnreachable(false);
      } catch {
        if (!cancelled) setUnreachable(true);
      }
    };
    void poll();
    const timer = setInterval(poll, pollMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [pollMs]);

  return { report, unreachable };
}

const DOT: Record<string, string> = {
  ok: "bg-emerald-400",
  degraded: "bg-amber-400",
  unhealthy: "bg-rose-500",
  unreachable: "bg-zinc-500",
};

/** A compact status pill; hover it for per-dependency detail. */
export function HealthIndicator({
  report,
  unreachable,
}: {
  report: HealthReport | null;
  unreachable: boolean;
}) {
  const status = unreachable || !report ? "unreachable" : report.status;
  const detail =
    report && !unreachable
      ? Object.entries(report.checks)
          .map(([name, probe]) => `${name}: ${probe.ok ? "ok" : probe.error}`)
          .join("\n")
      : "The API is not answering.";

  return (
    <div
      title={detail}
      className="flex shrink-0 items-center gap-2 rounded-full border border-zinc-800 px-3 py-1"
    >
      <span className={`h-2 w-2 rounded-full ${DOT[status]}`} />
      <span className="font-mono text-xs text-zinc-400">{status}</span>
    </div>
  );
}
