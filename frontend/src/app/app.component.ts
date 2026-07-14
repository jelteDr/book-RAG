import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';

/** Eine zitierte Quelle (aus dem `done`-Event des Backends). */
interface Source {
  marker: number;
  book_title: string | null;
  chapter: string | null;
  score: number;
  char_start: number | null;
  char_end: number | null;
  text: string;
}

interface Meta {
  model: string;
  ttft_ms: number;
  e2e_ms: number;
  tps: number;
}

interface Message {
  role: 'user' | 'assistant';
  text: string;
  sources?: Source[];
  meta?: Meta;
  error?: string;
}

/** Segment einer Antwort — Klartext oder ein anklickbarer [n]-Marker. */
interface Segment {
  text: string;
  marker: number | null;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="app">
      <header>
        <h1>book-RAG</h1>
        <div class="controls">
          <label>
            Modell
            <select [(ngModel)]="selectedModel">
              <option *ngFor="let m of models" [value]="m">{{ m }}</option>
            </select>
          </label>
          <label>
            Gruppe
            <input [(ngModel)]="groupId" placeholder="z. B. horror-classics" />
          </label>
        </div>
      </header>

      <main class="chat">
        <div *ngFor="let msg of messages" class="msg" [class.user]="msg.role === 'user'">
          <div class="bubble">
            <ng-container *ngIf="msg.role === 'assistant'; else plain">
              <span *ngFor="let seg of segments(msg.text)">
                <a *ngIf="seg.marker !== null; else t"
                   class="cite" (click)="toggleSource(msg, seg.marker)">[{{ seg.marker }}]</a>
                <ng-template #t>{{ seg.text }}</ng-template>
              </span>
              <span *ngIf="msg.error" class="error">{{ msg.error }}</span>

              <div class="sources" *ngIf="msg.sources?.length">
                <span class="src-title">Quellen:</span>
                <button *ngFor="let s of msg.sources" class="chip"
                        [class.active]="expanded === s"
                        (click)="expanded = expanded === s ? null : s">
                  [{{ s.marker }}] {{ s.chapter }} · {{ s.score }}
                </button>
              </div>
              <div class="src-detail" *ngIf="expanded && msg.sources?.includes(expanded)">
                <strong>[{{ expanded.marker }}] {{ expanded.book_title }} — {{ expanded.chapter }}</strong>
                <p>{{ expanded.text }}</p>
              </div>

              <div class="meta" *ngIf="msg.meta">
                {{ msg.meta.model }} · TTFT {{ msg.meta.ttft_ms }} ms ·
                {{ msg.meta.e2e_ms }} ms · {{ msg.meta.tps }} tok/s
              </div>
            </ng-container>
            <ng-template #plain>{{ msg.text }}</ng-template>
          </div>
        </div>
        <div *ngIf="streaming" class="typing">…</div>
      </main>

      <footer>
        <textarea [(ngModel)]="question" (keydown.enter)="onEnter($event)"
                  placeholder="Frage zum Buch stellen…" rows="2"></textarea>
        <button (click)="send()" [disabled]="streaming || !question.trim()">Senden</button>
      </footer>
    </div>
  `,
  styles: [`
    .app { display: flex; flex-direction: column; height: 100vh; max-width: 820px; margin: 0 auto; }
    header { padding: 12px 16px; border-bottom: 1px solid var(--border); }
    header h1 { margin: 0 0 8px; font-size: 18px; }
    .controls { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--muted); }
    .controls select, .controls input {
      margin-left: 6px; background: var(--panel-2); color: var(--text);
      border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px;
    }
    .chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { display: flex; }
    .msg.user { justify-content: flex-end; }
    .bubble { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 10px 14px; max-width: 90%; line-height: 1.5; white-space: pre-wrap; }
    .msg.user .bubble { background: var(--user); }
    .cite { color: var(--accent); cursor: pointer; font-weight: 600; }
    .cite:hover { text-decoration: underline; }
    .sources { margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
    .src-title { font-size: 12px; color: var(--muted); }
    .chip { background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
      border-radius: 999px; padding: 3px 10px; font-size: 12px; cursor: pointer; }
    .chip.active { border-color: var(--accent); color: var(--accent); }
    .src-detail { margin-top: 8px; background: var(--panel-2); border-left: 3px solid var(--accent);
      border-radius: 6px; padding: 8px 12px; font-size: 13px; }
    .src-detail p { margin: 6px 0 0; color: var(--muted); max-height: 180px; overflow-y: auto; }
    .meta { margin-top: 8px; font-size: 11px; color: var(--muted); }
    .error { color: #ff8080; }
    .typing { color: var(--muted); padding-left: 16px; }
    footer { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border); }
    footer textarea { flex: 1; resize: none; background: var(--panel-2); color: var(--text);
      border: 1px solid var(--border); border-radius: 8px; padding: 8px; font: inherit; }
    footer button { background: var(--accent); color: #0b1020; border: 0; border-radius: 8px;
      padding: 0 18px; font-weight: 600; cursor: pointer; }
    footer button:disabled { opacity: 0.5; cursor: default; }
  `],
})
export class AppComponent implements OnInit {
  models: string[] = [];
  selectedModel = '';
  groupId = 'horror-classics';
  question = '';
  messages: Message[] = [];
  streaming = false;
  expanded: Source | null = null;

  async ngOnInit(): Promise<void> {
    try {
      const resp = await fetch('/api/models');
      const data = await resp.json();
      this.models = data.available ?? [];
      this.selectedModel = data.default && this.models.includes(data.default)
        ? data.default
        : (this.models[0] ?? '');
    } catch {
      this.models = [];
    }
  }

  onEnter(event: Event): void {
    const ke = event as KeyboardEvent;
    if (!ke.shiftKey) {
      ke.preventDefault();
      this.send();
    }
  }

  /** Zerlegt eine Antwort in Klartext- und [n]-Marker-Segmente. */
  segments(text: string): Segment[] {
    const out: Segment[] = [];
    const re = /\[(\d+)\]/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) out.push({ text: text.slice(last, m.index), marker: null });
      out.push({ text: m[0], marker: Number(m[1]) });
      last = re.lastIndex;
    }
    if (last < text.length) out.push({ text: text.slice(last), marker: null });
    return out;
  }

  toggleSource(msg: Message, marker: number): void {
    const src = msg.sources?.find((s) => s.marker === marker) ?? null;
    this.expanded = this.expanded === src ? null : src;
  }

  async send(): Promise<void> {
    const q = this.question.trim();
    if (!q || this.streaming) return;
    this.messages.push({ role: 'user', text: q });
    const assistant: Message = { role: 'assistant', text: '' };
    this.messages.push(assistant);
    this.question = '';
    this.streaming = true;

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          model: this.selectedModel || null,
          group_id: this.groupId.trim() || null,
        }),
      });
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';
        for (const part of parts) this.handleEvent(part, assistant);
      }
    } catch (e) {
      assistant.error = 'Verbindung fehlgeschlagen: ' + e;
    } finally {
      this.streaming = false;
    }
  }

  private handleEvent(part: string, assistant: Message): void {
    let ev = '';
    let data = '';
    for (const line of part.split('\n')) {
      if (line.startsWith('event:')) ev = line.slice(6).trim();
      else if (line.startsWith('data:')) data = line.slice(5).trim();
    }
    if (!data) return;
    const payload = JSON.parse(data);
    if (ev === 'token') {
      assistant.text += payload.content;
    } else if (ev === 'done') {
      assistant.sources = payload.sources ?? [];
      assistant.meta = {
        model: payload.model,
        ttft_ms: payload.ttft_ms,
        e2e_ms: payload.e2e_ms,
        tps: payload.tps,
      };
    } else if (ev === 'error') {
      assistant.error = payload.message;
    }
  }
}
