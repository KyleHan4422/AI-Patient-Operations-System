# AI-Patient-Operations-System

## Guardrails: two checks no model is asked to make

| | G0 emergency filter | G5 booking policy |
|---|---|---|
| Where | The chat route, before anything else is resolved; with the conversation's last two messages once past the rate limit (`guardrails/emergency.py`) | `BookingDesk.book()`, before the calendar is called (`guardrails/booking_policy.py`) |
| What | Vocabulary rules from `knowledge_base/dental-emergencies.md`, words in either order: breathing or swallowing trouble, spreading or airway swelling, allergic reaction, bleeding that will not stop, face/jaw/head injury or fainting, chest pain, swelling with fever (-> call 911); thoughts of suicide or self-harm (-> 988, or 911 in danger); a knocked-out tooth (-> call the clinic now). In English, Spanish and Mandarin, the clinic's three languages; a Spanish or Mandarin match is answered in that language first | Right length for the procedure, inside the provider's hours, not a closure, on the slot grid, at least the lead time ahead, within the horizon |
| Answer | Fixed text with its source: no model, no graph, any length up to 20,000 characters (others stop at 2,000) | `INVALID`, which the booking path answers with fresh times |
| Cannot be stopped by | The rate limit, a missing model or embedder, Redis being down, a slow or failed transcript write (both writes share one 2 s budget; reported as `degraded: ["transcript"]`) | A database it cannot read: that is `TRANSIENT`, said as "the calendar did not answer" with the time kept for a retry |
| Leaves behind | Transcript row (`meta.guardrail = "G0"`), a checkpoint the next turn continues from, no open booking, and a Redis mark (R6) so that even if the checkpoint write fails, the booking path asks for the read-back again instead of taking the next "yes" -- within a write budget of its own (10 per client, one more per 10 s), past which the reply still goes out unrecorded, so an emergency keyword is not a way around R5 into the database | A `tool_calls` row, `error: invalid (G5:<rules>)` |
| Proven by | `evals/emergency/holdout.yaml`, never used to write a rule: **46/46 recalled, 1/23 negatives flagged**. `cases.yaml`, the development set: 112/112, 4/60 flagged (all labelled known false positives). `test_emergency_route.py` | `test_booking_policy.py`: every slot `compute_slots` generates passes, nudged ones do not |

Both fail in the safe direction on purpose. G0 does not understand negation --
"I'm not having trouble breathing" is told to call 911 -- because over-triage
costs a sentence and a miss does not bear thinking about. G5 re-checks even
what the booking path itself offered: a slot offered at exactly the lead time
and confirmed five minutes later is refused and re-offered, not quietly booked.
The one exception is a replay: a retried "yes" whose key already has a row is
passed to the calendar, which returns that row -- refusing it would tell a
booked patient they are not. Staff may book longer visits, off the grid and
inside the lead time.

G0's numbers deserve a caveat. Its first version spelled out phrases and
scored 50/50 on the set written alongside it -- and 12/29 on phrasings written
afterwards. The rules now match each emergency's vocabulary in either order,
and the holdout exists so that a number like that cannot happen silently
again. But the holdout was written by the rules' author; a clinician's list is
the next test it needs, and the Spanish and Mandarin replies want a native
speaker's review. Known limits:

- An emergency told in pieces is read across the last two messages, and only
  counts if the newest one adds something -- so "ok thanks" is not answered
  with the same reply again. That read is after the rate limit (it reads the
  database): a client over its limit is refused before it can happen. And it
  over-triages across messages: "how long does swelling last?" then "can I
  use my eye drops?" is the holdout's one false positive.
- Past 20,000 characters, the request is refused before G0 reads it.
- If the checkpoint write and Redis both fail, a booking in progress stays
  open.
- Three languages, by keyword. Anything else reaches the classifier, and the
  knowledge branch -- which can quote the emergencies document -- is the
  backstop.

## Booking: what the model may do, and what only code does

A patient books in chat: *book me a cleaning* -> phone number -> three offered
times -> *the second one* -> a read-back -> *yes* -> **You're booked**. The
language model only reads each message into a typed proposal
(`agents/booking.py`); it has no tools, and it picks a time only by the number
it was offered. Everything that changes anything is deterministic code in
`graph/nodes/booking.py`:

| Step | What happens | Guarantee |
|---|---|---|
| Who | Look the patient up by phone, or by name and date of birth | Two patients with one name are told apart by asking; nobody unknown is booked or created |
| Offer | Find free slots, hold three for 120 s (earliest per day) | Another conversation is not offered a held slot |
| Read back | "Adult cleaning with Dr. Chen on Fri 02 Oct at 09:00. Shall I book it?" | Nothing is written before an explicit yes to this question |
| Execute | One write, through the breaker and in-flight dedup | Idempotency key = thread + patient + slot, so a repeated yes makes one appointment |
| Verify | Read the appointment back by that key | "You're booked" is worded from the row, with its reference; a write that cannot be read back is reported as uncertain |

A taken slot is re-offered, an open breaker or a timed-out write is said
plainly (and a yes retries with the same key), and changing or cancelling an
appointment is referred to the front desk. `api/tests/test_booking_flow.py`
drives each case against a real Postgres and Redis.

## Redis: what it does, and what happens without it

Correctness lives in Postgres; coordination lives in Redis. Every guarantee
about bookings -- no two appointments overlap, one request makes one
appointment -- is a Postgres constraint. Redis makes the system smoother, and
losing all of it makes the system slower and a little clumsier, never wrong.
R1, R3 and R4 sit on the booking path; R5 is in front of every chat turn.
`/health` says so too: Redis down is `degraded` (200), never `unhealthy`.

| Capability | What it does | In use | With Redis down | Correctness | Proven by |
|---|---|---|---|---|---|
| R1 slot holds | An offered slot is held for 120 s for the conversation it was offered to, per 30-minute grid cell so overlapping slots collide | Now: offering slots | Offer without holding; the `no_overlap` EXCLUDE constraint decides at booking | Unaffected -- more conflicts | `test_holds_degrade_to_offering_without_a_hold` |
| R3 circuit breaker | After 5 consecutive calendar failures, fail fast for 30 s, then let exactly one probe through -- shared by every worker | Now: every calendar call | Per-process breaker | Unaffected -- weaker protection | `test_breaker_degrades_to_a_per_process_breaker_that_still_opens` |
| R4 in-flight dedup | A duplicate request waits for the original instead of calling the calendar again | Now: writing a booking | Both requests write; the UNIQUE idempotency key returns one appointment | Unaffected -- one extra call | `test_dedup_degrades_to_the_unique_key` |
| R5 rate limit | Token bucket per client on chat turns; `429` + `Retry-After` | Now: `POST /api/chat/turn` | Let the request through | Unaffected | `test_rate_limit_fails_open` |
| R6 emergency marks | "This conversation reported an emergency at T", read before a booking is written | Now: G0 turns and the booking read-back | No mark; the checkpoint write alone ends the booking | Unaffected unless that write also fails | `test_if_the_checkpoint_write_fails_the_mark_still_stops_the_yes` |

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
