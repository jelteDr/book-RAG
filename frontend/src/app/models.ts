/** Datentypen der book-RAG-API. */

export interface Source {
  marker: number;
  book_title: string | null;
  chapter: string | null;
  score: number;
  char_start: number | null;
  char_end: number | null;
  text: string;
}

export interface Meta {
  model: string;
  ttft_ms: number;
  e2e_ms: number;
  tps: number;
}

export interface Message {
  role: 'user' | 'assistant';
  text: string;
  sources?: Source[];
  meta?: Meta;
  error?: string;
}

export interface ChatRequest {
  question: string;
  model: string | null;
  group_id: string | null;
}

export interface ModelsResponse {
  default: string;
  available: string[];
}

/** Ein geparstes SSE-Event des /chat-Streams. */
export interface ChatEvent {
  event: 'token' | 'done' | 'error';
  data: {
    content?: string;
    sources?: Source[];
    model?: string;
    ttft_ms?: number;
    e2e_ms?: number;
    tps?: number;
    message?: string;
  };
}
