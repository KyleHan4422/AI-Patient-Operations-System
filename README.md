# AI-Patient-Operations-System
## Redis: what it does, and what happens without it

Correctness lives in Postgres; coordination lives in Redis. Every guarantee
about bookings -- no two appointments overlap, one request makes one
appointment -- is a Postgres constraint. Redis makes the system smoother, and
losing all of it makes the system slower and a little clumsier, never wrong.
R1, R3 and R4 are built and tested here; the booking path that calls them is
Phase 6. R5 is already in front of every chat turn.
`/health` says so too: Redis down is `degraded` (200), never `unhealthy`.

| Capability | What it does | In use | With Redis down | Correctness | Proven by |
|---|---|---|---|---|---|
| R1 slot holds | An offered slot is held for 120 s for the conversation it was offered to, per 30-minute grid cell so overlapping slots collide | Phase 6 (offering slots) | Offer without holding; the `no_overlap` EXCLUDE constraint decides at booking | Unaffected -- more conflicts | `test_holds_degrade_to_offering_without_a_hold` |
| R3 circuit breaker | After 5 consecutive calendar failures, fail fast for 30 s, then let exactly one probe through -- shared by every worker | Phase 6 (around the calendar) | Per-process breaker | Unaffected -- weaker protection | `test_breaker_degrades_to_a_per_process_breaker_that_still_opens` |
| R4 in-flight dedup | A duplicate request waits for the original instead of calling the calendar again | Phase 6 (writing a booking) | Both requests write; the UNIQUE idempotency key returns one appointment | Unaffected -- one extra call | `test_dedup_degrades_to_the_unique_key` |
| R5 rate limit | Token bucket per client on chat turns; `429` + `Retry-After` | Now: `POST /api/chat/turn` | Let the request through | Unaffected | `test_rate_limit_fails_open` |

Every fallback is visible: it is logged at WARN as `degraded_mode`, recorded in
the turn's `degraded_modes`, and returned in the chat stream's `done` event
(`"degraded": ["rate_limit"]`). A silent fallback is how a degraded system runs
for a month before anyone notices. A Redis that answers but refuses writes (MISCONF, OOM, READONLY) counts as
down, like one that does not answer; and after a failure, every feature skips
Redis for a few seconds (`REDIS_DOWN_BACKOFF_S`) rather than each call waiting
out its own timeout. `FAULT_INJECT=redis:unavailable` reproduces
the outage with Redis still running; `api/tests/test_degradation.py` runs
every row both that way and against a port nothing listens on.

What Redis is deliberately **not** used for:

| Not this | Why |
|---|---|
| Conversation state | The Postgres checkpointer already holds it; two state stores drift |
| Caching LLM responses | Little saved at this scale, and a cache-invalidation problem gained |
| Event log | The `tool_calls` table is already a queryable log, and it does not expire |
| System of record | Redis can lose data by design; nothing correct may depend on it |
