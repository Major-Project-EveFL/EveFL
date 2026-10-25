// Fetch helpers with a timeout and a readable error; the page decides how to show it.
export async function getJSON(url, { timeout = 20000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { signal: controller.signal, headers: { Accept: "application/json" } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);
    return data;
  } catch (err) {
    if (err.name === "AbortError") throw new Error("the server took too long to answer");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// Server-sent events. Returns a function that closes the stream. `onError` fires once; the stream is not retried automatically.
export function stream(url, { onEvent, onError }) {
  const source = new EventSource(url);
  let finished = false;
  const finish = () => { finished = true; source.close(); };
  for (const name of ["result", "done"]) {
    source.addEventListener(name, (event) => {
      try { onEvent(name, JSON.parse(event.data)); } catch (err) { finish(); onError(err); }
      if (name === "done") finish();
    });
  }
  source.onerror = () => { if (finished) return; finish(); onError(new Error("the connection to the demo server was lost")); };
  return finish;
}
