/** Datentypen der book-RAG-API. */

export interface Source {
  marker: number;
  book_title: string | null;
  chapter: string | null;
  score: number;
  char_start: number | null;
  char_end: number | null;
  text: string;
  supported?: boolean; // optionaler Faithfulness-Check (NLI): stützt der Chunk die Aussage?
  support_score?: number;
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
  conversation_id?: number | null;
}

export interface Conversation {
  id: number;
  title: string;
  group_id: string | null;
  updated_at: string;
}

export interface ConversationDetail {
  id: number;
  title: string;
  group_id: string | null;
  messages: { role: 'user' | 'assistant'; content: string; model: string | null; sources: Source[] }[];
}

export interface ModelsResponse {
  default: string;
  available: string[];
}

export interface Group {
  id: number;
  slug: string;
  name: string;
  kind: string | null;
  description: string | null;
  n_books?: number;
}

export interface Collection {
  id: number;
  slug: string;
  name: string;
  members: { slug: string; name: string }[];
  n_books: number;
}

export interface Book {
  id: number;
  book_key: string;
  title: string | null;
  author: string | null;
  language: string | null;
  n_chunks: number;
  status: string;
}

export interface ModelInfo {
  name: string;
  family: string | null;
  parameter_size: string | null;
  parameter_count: number | null;
  quantization: string | null;
  context_length: number | null;
  cutoff_date: string | null;
  source: string | null;
}

export interface MetricsRow {
  model: string;
  queries: number;
  avg_ttft_ms: number | null;
  avg_tps: number | null;
  avg_completion_tokens: number | null;
}

export interface CleaningReport {
  original_chars: number;
  cleaned_chars: number;
  lines_merged: number;
  dropped_ratio: number;
  warnings: string[];
  validation_errors?: string[];
  before_sample: string;
  after_sample: string;
}

export interface UploadResult {
  committed: boolean;
  title?: string | null;
  author?: string | null;
  report: CleaningReport;
  book?: Book;
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
    conversation_id?: number;
  };
}
