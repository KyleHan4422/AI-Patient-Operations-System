"use client";

import { useEffect, useState } from "react";
import { DegradedBanner } from "@/components/DegradedBanner";
import { API_BASE_URL, fetchHealth, type HealthReport } from "@/lib/health";

const POLL_MS = 3000;

const STATUS_STYLES: Record<string, { dot: string; label: string }> = {
  ok: { dot: "bg-emerald-400", label: "text-emerald-300" },
  degraded: { dot: "bg-amber-400", label: "text-amber-300" },
  unhealthy: { dot: "bg-rose-500", label: "text-rose-300" },
  unreachable: { dot: "bg-zinc-500", label: "text-zinc-400" },
};

export default function Home() {
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
        // Network-level failure: the API itself is not answering. This is a
        // different condition from `unhealthy`, which the API reports itself.
        if (!cancelled) setUnreachable(true);
      }
    };

    void poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const status = unreachable ? "unreachable" : (report?.status ?? "unreachable");
  const style = STATUS_STYLES[status];

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col gap-6 px-4 py-16">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight">
          Patient Operations System
        </h1>
        <p className="text-sm text-zinc-400">
          Phase 0 — infrastructure health. No clinical functionality yet.
        </p>
      </header>

      <section className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-4">
        <span className={`h-2.5 w-2.5 rounded-full ${style.dot}`} />
        <span className={`font-mono text-sm font-medium ${style.label}`}>
          {status}
        </span>
        {report && (
          <span className="ml-auto font-mono text-xs text-zinc-500">
            v{report.version}
          </span>
        )}
      </section>

      {report && <DegradedBanner degraded={report.degraded} />}

      {unreachable && (
        <div
          role="status"
          className="rounded-lg border border-zinc-700 bg-zinc-900/50 px-4 py-3 text-sm text-zinc-400"
        >
          Cannot reach the API at{" "}
          <code className="font-mono text-zinc-300">{API_BASE_URL}</code>. Is{" "}
          <code className="font-mono text-zinc-300">make dev</code> running?
        </div>
      )}

      {report && (
        <section className="flex flex-col gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Dependencies
          </h2>
          <ul className="divide-y divide-zinc-800 overflow-hidden rounded-lg border border-zinc-800">
            {Object.entries(report.checks).map(([name, probe]) => (
              <li
                key={name}
                className="flex items-baseline gap-3 bg-zinc-900/50 px-4 py-3"
              >
                <span
                  className={`h-2 w-2 shrink-0 translate-y-1 rounded-full ${
                    probe.ok ? "bg-emerald-400" : "bg-rose-500"
                  }`}
                />
                <span className="font-mono text-sm">{name}</span>
                <span className="ml-auto font-mono text-xs text-zinc-500">
                  {probe.ok ? `${probe.latency_ms} ms` : probe.error}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
