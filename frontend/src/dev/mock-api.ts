/**
 * Dev-only-Vorschaumodus (?mock=1): beantwortet fetch('/api/*') mit Beispieldaten, damit sich
 * Layout, Themes und Zustände ohne Backend/Ollama prüfen lassen. Wird nur unter ngDevMode geladen
 * (siehe main.ts) und landet nicht im Produktions-Build. Die Daten sind absichtlich sperrig
 * (lange Namen, große Zahlen), damit Überläufe auffallen.
 */
const LONG = 'Die außerordentlich lange und umständlich betitelte Gesamtausgabe sämtlicher Schauerromane';

const groups = [
  { id: 1, slug: 'Horror', name: 'Horror', kind: 'genre', description: null, n_books: 1 },
  { id: 2, slug: 'GoT', name: 'A Song of Ice and Fire', kind: 'franchise', description: null, n_books: 5 },
  { id: 3, slug: 'lang', name: LONG, kind: null, description: null, n_books: 2 },
];

const books: Record<number, unknown[]> = {
  1: [{ id: 1, book_key: 'dracula', title: 'Dracula', author: 'Bram Stoker', language: 'en', n_chunks: 581, status: 'ready' }],
  2: [1, 2, 3, 4, 5].map((n) => ({
    id: 10 + n, book_key: `got-${n}`, title: `A Song of Ice and Fire – Band ${n}`, author: 'George R. R. Martin',
    language: 'en', n_chunks: 600 + n * 37, status: 'ready',
  })),
  3: [
    { id: 31, book_key: 'lang-1', title: `${LONG}, Band 1 (kommentierte Neuausgabe)`, author: null, language: 'de', n_chunks: 1234567, status: 'ready' },
    { id: 32, book_key: 'ein_sehr_langer_dateiname_ohne_leerzeichen_der_nicht_umbrechen_kann.txt', title: null, author: 'Anonym', language: 'de', n_chunks: 12, status: 'ready' },
  ],
};

const collections = [
  { id: 1, slug: 'fantasy', name: 'Fantasy', members: [{ slug: 'GoT', name: 'A Song of Ice and Fire' }, { slug: 'lang', name: LONG }], n_books: 7 },
];

const modelNames = ['qwen2.5:7b-instruct-q4_k_m', 'gemma4:latest', 'mistral:latest', 'qwen3:0.6b', 'hf.co/irgendein-nutzer/sehr-langer-modellname-GGUF:Q5_K_M'];

const modelsInfo = modelNames.map((name, i) => ({
  name, family: 'qwen2', parameter_size: ['7.6B', '12B', '7.2B', '752M', '70.6B'][i], parameter_count: 7_600_000_000,
  quantization: 'Q4_K_M', context_length: [32768, 131072, 32768, 40960, 1048576][i],
  cutoff_date: i === 0 ? '2024-06-01' : null, source: 'ollama',
}));

const metrics = {
  per_model: [
    { model: modelNames[0], queries: 1284, avg_ttft_ms: 9358, avg_tps: 11.8, avg_completion_tokens: 212 },
    { model: modelNames[1], queries: 17, avg_ttft_ms: 55120, avg_tps: 6.2, avg_completion_tokens: 388 },
    { model: modelNames[2], queries: 4, avg_ttft_ms: 7420, avg_tps: 13.1, avg_completion_tokens: 97 },
    { model: modelNames[4], queries: 1, avg_ttft_ms: 147000, avg_tps: 0.9, avg_completion_tokens: 12 },
  ],
};

const sources = [
  { marker: 1, book_title: 'Dracula', chapter: 'CHAPTER II — Jonathan Harker’s Journal (continued)', score: 0.71, char_start: 100, char_end: 900, text: 'His face was a strong—a very strong—aquiline, with high bridge of the thin nose and peculiarly arched nostrils …', supported: true },
  { marker: 2, book_title: 'Dracula', chapter: 'CHAPTER III', score: 0.64, char_start: 1000, char_end: 1800, text: 'What manner of man is this, or what manner of creature is it in the semblance of man?', supported: false },
  { marker: 3, book_title: 'Dracula', chapter: 'CHAPTER I', score: 0.58, char_start: 5, char_end: 700, text: '' },
];

const answer =
  'Der Graf wird als großer alter Mann mit markanter Adlernase beschrieben [1]. Harker zweifelt später, ob er überhaupt ein Mensch ist [2].\n\nSeine Ankunft im Schloss steht am Anfang des Tagebuchs [3].';

const conversations = [
  { id: 1, title: 'Wie wird Graf Dracula beschrieben?', group_id: 'Horror', updated_at: '2026-09-20T10:00:00' },
  { id: 2, title: 'Eine ausgesprochen lange Frage, die als Chat-Titel garantiert nicht in die Seitenleiste passt', group_id: 'Horror', updated_at: '2026-09-19T10:00:00' },
  { id: 3, title: 'Wer sitzt auf dem Eisernen Thron?', group_id: 'GoT', updated_at: '2026-09-18T10:00:00' },
  { id: 4, title: 'Chat ohne Gruppe', group_id: null, updated_at: '2026-09-17T10:00:00' },
];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function chatStream(): Response {
  const encoder = new TextEncoder();
  const send = (event: string, data: unknown) => encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  const tokens = answer.split(/(?<= )/);
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      await new Promise((resolve) => setTimeout(resolve, 2500)); // Denk-Phase sichtbar machen
      for (const content of tokens) {
        controller.enqueue(send('token', { content }));
        await new Promise((resolve) => setTimeout(resolve, 40));
      }
      controller.enqueue(send('done', { sources, model: modelNames[0], ttft_ms: 2500, e2e_ms: 4200, tps: 11.8, conversation_id: 1 }));
      controller.close();
    },
  });
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
}

function respond(path: string, method: string): Response {
  if (path === '/api/chat') return chatStream();
  if (method !== 'GET') {
    if (path.endsWith('/upload')) {
      return json({
        committed: false, title: 'Dracula', author: 'Bram Stoker',
        report: {
          original_chars: 890000, cleaned_chars: 861234, lines_merged: 15321, dropped_ratio: 0.032,
          warnings: ['Kein Gutenberg-Header gefunden'], validation_errors: ['Buchstabenanteil unter 0,5'],
          before_sample: '', after_sample: 'Jonathan Harker’s Journal. 3 May. Bistritz.—Left Munich at 8:35 P. M.',
        },
      });
    }
    return json({});
  }
  if (path === '/api/models') return json({ default: modelNames[0], available: modelNames });
  if (path === '/api/models/info') return json(modelsInfo);
  if (path === '/api/metrics') return json(metrics);
  if (path === '/api/groups') return json(groups);
  if (path === '/api/collections') return json(collections);
  if (path === '/api/conversations') return json(conversations);
  const conv = path.match(/^\/api\/conversations\/(\d+)$/);
  if (conv) {
    const found = conversations.find((c) => c.id === Number(conv[1]))!;
    return json({
      ...found,
      messages: [
        { role: 'user', content: found.title, model: null, sources: [] },
        { role: 'assistant', content: answer, model: modelNames[0], sources },
      ],
    });
  }
  const groupBooks = path.match(/^\/api\/groups\/(\d+)\/books$/);
  if (groupBooks) return json(books[Number(groupBooks[1])] ?? []);
  return json({ detail: 'mock: unbekannter Pfad' }, 404);
}

export function installMockApi(): void {
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url, location.origin);
    if (!url.pathname.startsWith('/api/')) return realFetch(input, init);
    return respond(url.pathname, (init?.method ?? 'GET').toUpperCase());
  };
  console.info('[mock] /api/* wird mit Beispieldaten beantwortet – ?mock=0 beendet den Modus.');
}
