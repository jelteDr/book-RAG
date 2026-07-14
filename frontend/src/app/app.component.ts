import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ChatService } from './chat.service';
import { Message, Source } from './models';

/** Ein Antwort-Segment: Klartext oder ein anklickbarer [n]-Marker. */
interface Segment {
  text: string;
  marker: number | null;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {
  private readonly chat = inject(ChatService);

  readonly models = signal<string[]>([]);
  readonly selectedModel = signal('');
  readonly groupId = signal('horror-classics');
  readonly question = signal('');
  readonly messages = signal<Message[]>([]);
  readonly draft = signal('');
  readonly streaming = signal(false);
  readonly expanded = signal<Source | null>(null);

  constructor() {
    void this.loadModels();
  }

  private async loadModels(): Promise<void> {
    try {
      const res = await this.chat.getModels();
      const available = res.available ?? [];
      this.models.set(available);
      this.selectedModel.set(
        res.default && available.includes(res.default) ? res.default : (available[0] ?? ''),
      );
    } catch {
      this.models.set([]);
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

  onCite(message: Message, marker: number): void {
    const source = message.sources?.find((s) => s.marker === marker);
    if (source) this.toggleSource(source);
  }

  toggleSource(source: Source): void {
    this.expanded.update((cur) => (cur === source ? null : source));
  }

  onEnter(event: Event): void {
    const ke = event as KeyboardEvent;
    if (!ke.shiftKey) {
      ke.preventDefault();
      void this.send();
    }
  }

  async send(): Promise<void> {
    const question = this.question().trim();
    if (!question || this.streaming()) return;

    this.messages.update((list) => [...list, { role: 'user', text: question }]);
    this.question.set('');
    this.draft.set('');
    this.streaming.set(true);

    try {
      const request = {
        question,
        model: this.selectedModel() || null,
        group_id: this.groupId().trim() || null,
      };
      for await (const ev of this.chat.stream(request)) {
        if (ev.event === 'token') {
          this.draft.update((d) => d + (ev.data.content ?? ''));
        } else if (ev.event === 'done') {
          this.finish({
            role: 'assistant',
            text: this.draft(),
            sources: ev.data.sources ?? [],
            meta: {
              model: ev.data.model ?? '',
              ttft_ms: ev.data.ttft_ms ?? 0,
              e2e_ms: ev.data.e2e_ms ?? 0,
              tps: ev.data.tps ?? 0,
            },
          });
        } else if (ev.event === 'error') {
          this.finish({ role: 'assistant', text: this.draft(), error: ev.data.message });
        }
      }
    } catch (e) {
      this.finish({ role: 'assistant', text: this.draft(), error: `Verbindung fehlgeschlagen: ${e}` });
    } finally {
      this.streaming.set(false);
      this.draft.set('');
    }
  }

  private finish(message: Message): void {
    this.messages.update((list) => [...list, message]);
  }
}
