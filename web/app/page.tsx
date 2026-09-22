"use client";

import { Chat } from "@/components/Chat";
import { DegradedBanner } from "@/components/DegradedBanner";
import { HealthIndicator, useHealth } from "@/components/HealthIndicator";

export default function Home() {
  const { report, unreachable } = useHealth();

  return (
    <main className="mx-auto flex h-dvh max-w-2xl flex-col gap-4 px-4 py-6">
      <header className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold tracking-tight">Patient Operations System</h1>
          <p className="text-sm text-zinc-400">
            Phase 2 — conversation memory only. No clinic data yet.
          </p>
        </div>
        <HealthIndicator report={report} unreachable={unreachable} />
      </header>

      {report && <DegradedBanner degraded={report.degraded} />}

      <Chat />
    </main>
  );
}
