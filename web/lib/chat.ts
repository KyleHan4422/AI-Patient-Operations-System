/**
 * Client for the chat API. The wire protocol is documented in
 * api/src/patient_ops/api/routes_chat.py:
 *
 *   meta -> stage -> token* -> done     (success)
 *   meta -> stage -> token* -> error    (failure after the stream started)
 *
 * An emergency (guardrail G0) is answered before anything else runs: stage
 * "emergency", then a done that carries `guardrail`. It is never refused with
 * a 429 or a 503, so the client need not special-case those for it.
 *
 * Most replies carry no tokens at all. An answer about the clinic is checked
 * against the passages it cites before it is said, so it arrives whole, in
 * `done`; `stage` is what the client shows meanwhile.
 *
 * Failures before the stream starts (bad input, no model configured) come back
 * as ordinary HTTP errors instead, and are reported through the same onError.
 */

import { API_BASE_URL } from "@/lib/health";
import { readSse } from "@/lib/sse";

export type Role = "user" | "assistant";
/** `guardrail` is "G0" on a fixed emergency reply, shown differently. */
export type ChatMessage = { role: Role; content: string; guardrail?: string };

export type TurnError = { code: string; message: string; requestId: string };

export type TurnHandlers = {
  onMeta: (threadId: string) => void;
  /** Which branch the turn took: "knowledge", "booking", "smalltalk" or "emergency". */
  onStage: (intent: string) => void;
  onToken: (text: string) => void;
  onDone: (text: string, guardrail?: string) => void;
  onError: (error: TurnError) => void;
};

export async function streamTurn(
  message: string,
  threadId: string | null,
  handlers: TurnHandlers,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/chat/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId ?? undefined, channel: "web" }),
    });
  } catch {
    handlers.onError({
      code: "unreachable",
      message: `Cannot reach the API at ${API_BASE_URL}. Is \`make dev\` running?`,
      requestId: "",
    });
    return;
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    handlers.onError({
      code: String(response.status),
      message:
        typeof detail?.message === "string"
          ? detail.message
          : `The request was rejected (HTTP ${response.status}).`,
      requestId: response.headers.get("x-request-id") ?? "",
    });
    return;
  }

  let finished = false;
  try {
    await readSse(response, ({ event, data }) => {
      const payload = JSON.parse(data);
      if (event === "meta") handlers.onMeta(payload.thread_id);
      else if (event === "stage") handlers.onStage(payload.intent);
      else if (event === "token") handlers.onToken(payload.text);
      else if (event === "done") {
        finished = true;
        handlers.onDone(payload.text, payload.guardrail?.id);
      } else if (event === "error") {
        finished = true;
        handlers.onError({
          code: payload.code,
          message: payload.message,
          requestId: payload.request_id,
        });
      }
    });
  } catch {
    // fall through: the connection dropped mid-stream
  }
  if (!finished) {
    handlers.onError({
      code: "interrupted",
      message: "The connection closed before the reply finished.",
      requestId: "",
    });
  }
}

/** The conversation so far, or null if the server has no such thread. */
export async function fetchHistory(threadId: string): Promise<ChatMessage[] | null> {
  const response = await fetch(
    `${API_BASE_URL}/api/chat/threads/${encodeURIComponent(threadId)}/messages`,
    { cache: "no-store" },
  );
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`history request failed (HTTP ${response.status})`);
  const rows: { role: Role; content: string; guardrail: string | null }[] =
    await response.json();
  return rows.map(({ role, content, guardrail }) => ({
    role,
    content,
    ...(guardrail ? { guardrail } : {}),
  }));
}

// The thread id is kept in localStorage so a page reload resumes the same
// conversation. Storage can be unavailable (private windows, blocked site
// data): every access is guarded, and the page still works without it -- it
// just starts a new conversation on reload.
const THREAD_KEY = "patient-ops.thread-id";

export function loadThreadId(): string | null {
  try {
    return window.localStorage.getItem(THREAD_KEY);
  } catch {
    return null;
  }
}

export function saveThreadId(threadId: string | null): void {
  try {
    if (threadId) window.localStorage.setItem(THREAD_KEY, threadId);
    else window.localStorage.removeItem(THREAD_KEY);
  } catch {
    // ignore: see loadThreadId
  }
}
