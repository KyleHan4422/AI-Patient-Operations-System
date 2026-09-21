/**
 * Degradation must be visible. A system that silently drops to a fallback path
 * is how production incidents stay undiscovered -- so every degraded mode the
 * backend reports gets rendered here. Phase 10 reuses this on /ops.
 */
export function DegradedBanner({ degraded }: { degraded: string[] }) {
  if (degraded.length === 0) return null;
  return (
    <div
      role="status"
      className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200"
    >
      <span className="font-semibold">Degraded mode</span>
      {" — "}
      {degraded.join(", ")} unavailable. Requests still succeed; coordination
      features are running on their fallback paths.
    </div>
  );
}
