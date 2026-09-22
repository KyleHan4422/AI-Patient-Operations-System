/**
 * Minimal Server-Sent Events reader for a fetch() response.
 *
 * The browser's EventSource cannot be used: it only issues GET requests, and a
 * chat turn is a POST carrying the message. So the stream is read by hand.
 *
 * Network chunks do not line up with events -- one read can end halfway
 * through an event -- so text is buffered until a blank line closes an event.
 */

export type SseEvent = { event: string; data: string };

export async function readSse(
  response: Response,
  onEvent: (event: SseEvent) => void,
): Promise<void> {
  if (!response.body) throw new Error("response has no body");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = (buffer + value).replace(/\r\n/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const parsed = parseBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (parsed) onEvent(parsed);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line === "" || line.startsWith(":")) continue; // ": ping" keep-alives
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length > 0 ? { event, data: data.join("\n") } : null;
}
