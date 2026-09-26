"use client";

import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  type ChatMessage,
  type TurnError,
  fetchHistory,
  loadThreadId,
  saveThreadId,
  streamTurn,
} from "@/lib/chat";

const MAX_MESSAGE_CHARS = 2000; // mirrors the API's limit

// What the wait means. An answer about the clinic is checked against its
// evidence before it is said, so nothing streams and the pause is real. Saying
// what is happening beats a spinner, and beats showing a price that is then
// taken back.
const STAGE_LABELS: Record<string, string> = {
  knowledge: "Looking that up in the clinic's records…",
  booking: "One moment…",
  smalltalk: "…",
  emergency: "…",
};

// 911, 988 and the clinic's number in an emergency reply become tel: links,
// so a patient on a phone is one tap from calling.
const PHONE_NUMBER = /(\b911\b|\b988\b|\(\d{3}\) \d{3}-\d{4})/;

export function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // The reply while it streams in. Provisional: replaced by `done.text`.
  const [streaming, setStreaming] = useState<string | null>(null);
  // Which branch the turn took, while it is still running.
  const [stage, setStage] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<TurnError | null>(null);
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  // Resume the stored conversation. The transcript comes from the server's
  // messages table; the model's memory of it lives in the checkpoint.
  useEffect(() => {
    const stored = loadThreadId();
    if (!stored) return;
    setThreadId(stored);
    let cancelled = false;
    fetchHistory(stored)
      .then((history) => {
        if (cancelled) return;
        if (history === null) {
          // The server no longer knows this thread (e.g. after make db-reset).
          saveThreadId(null);
          setThreadId(null);
        } else {
          setMessages(history);
        }
      })
      .catch(() => {
        // API down: keep the thread id so the conversation resumes once it
        // is back. The health pill shows what is wrong.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, streaming, error]);

  async function send(text: string) {
    const message = text.trim();
    // Sending is disabled while a reply streams: two turns racing on one
    // thread would interleave. (Server-side serialisation comes in Phase 5.)
    if (!message || busy) return;

    setBusy(true);
    setError(null);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: message }]);
    setStreaming("");
    setStage(null);

    await streamTurn(message, threadId, {
      onMeta: (id) => {
        setThreadId(id);
        saveThreadId(id);
      },
      onStage: (intent) => setStage(intent),
      onToken: (token) => setStreaming((s) => (s ?? "") + token),
      onDone: (final, guardrail) => {
        setStreaming(null);
        setStage(null);
        setMessages((m) => [
          ...m,
          { role: "assistant", content: final, ...(guardrail ? { guardrail } : {}) },
        ]);
      },
      onError: (turnError) => {
        setStreaming(null);
        setStage(null);
        setError(turnError);
        // The turn did not complete: take the message back so it can be resent.
        setMessages((m) => m.slice(0, -1));
        setInput(message);
      },
    });
    setBusy(false);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter adds a line. Not while an input method is
    // composing: there Enter confirms the candidate (typing Chinese, say).
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send(input);
    }
  }

  function startOver() {
    saveThreadId(null);
    setThreadId(null);
    setMessages([]);
    setError(null);
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/40">
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
        <span className="font-mono text-xs text-zinc-500">
          {threadId ? `thread ${threadId.slice(0, 8)}` : "new conversation"}
        </span>
        <button
          type="button"
          onClick={startOver}
          disabled={busy || (!threadId && messages.length === 0)}
          className="text-xs text-zinc-400 hover:text-zinc-100 disabled:opacity-40"
        >
          New conversation
        </button>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4" aria-live="polite">
        {messages.length === 0 && streaming === null && (
          <p className="text-sm text-zinc-500">
            Ask about opening hours, prices, which insurance we take, or the
            clinic&apos;s policies. Answers say where they came from — and when the
            records don&apos;t cover something, so does that.
          </p>
        )}
        {messages.map((message, i) => (
          <Bubble
            key={i}
            role={message.role}
            text={message.content}
            emergency={message.guardrail === "G0"}
          />
        ))}
        {streaming !== null && (
          <Bubble
            role="assistant"
            text={streaming || (stage ? (STAGE_LABELS[stage] ?? "…") : "…")}
            pending
          />
        )}
        {error && (
          <div
            role="alert"
            className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200"
          >
            {error.message}{" "}
            <span className="font-mono text-xs text-rose-300/70">
              ({error.code}
              {error.requestId && ` · ${error.requestId}`})
            </span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void send(input);
        }}
        className="flex gap-2 border-t border-zinc-800 p-3"
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          maxLength={MAX_MESSAGE_CHARS}
          disabled={busy}
          placeholder="Type a message…"
          aria-label="Message"
          className="flex-1 resize-none rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm outline-none focus:border-zinc-500 disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="self-end rounded-md bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900 disabled:opacity-40"
        >
          {busy ? "…" : "Send"}
        </button>
      </form>
    </section>
  );
}

function Bubble({
  role,
  text,
  pending = false,
  emergency = false,
}: {
  role: string;
  text: string;
  pending?: boolean;
  emergency?: boolean;
}) {
  const mine = role === "user";
  const tone = mine
    ? "bg-sky-600 text-white"
    : emergency
      ? "border border-rose-500/60 bg-rose-950/60 text-rose-50"
      : "bg-zinc-800 text-zinc-100";
  return (
    <div className={`flex ${mine ? "justify-end" : "justify-start"}`}>
      <div
        role={emergency ? "alert" : undefined}
        className={`max-w-[85%] whitespace-pre-wrap break-words rounded-2xl px-3.5 py-2 text-sm leading-relaxed ${tone} ${
          pending ? "opacity-80" : ""
        }`}
      >
        {emergency ? <Linked text={text} /> : text}
      </div>
    </div>
  );
}

function Linked({ text }: { text: string }) {
  // split() with a capturing group keeps the numbers, at the odd indices.
  return text.split(PHONE_NUMBER).map((part, i) =>
    i % 2 === 1 ? (
      <a
        key={i}
        href={`tel:${part.replace(/\D/g, "")}`}
        className="font-semibold underline underline-offset-2"
      >
        {part}
      </a>
    ) : (
      part
    ),
  );
}
