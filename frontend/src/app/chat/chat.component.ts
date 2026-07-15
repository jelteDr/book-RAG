import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ChatService } from '../chat.service';
import { LibraryService } from '../library.service';
import { Conversation, Group, Message, Source } from '../models';

interface Segment {
  text: string;
  marker: number | null;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './chat.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChatComponent {
  private readonly chat = inject(ChatService);
  private readonly library = inject(LibraryService);

  readonly models = signal<string[]>([]);
  readonly selectedModel = signal('');
  readonly groups = signal<Group[]>([]);
  readonly selectedGroup = signal('');
  readonly conversations = signal<Conversation[]>([]);
  readonly currentConversationId = signal<number | null>(null);
  readonly question = signal('');
  readonly messages = signal<Message[]>([]);
  readonly draft = signal('');
  readonly streaming = signal(false);
  readonly expanded = signal<Source | null>(null);

  constructor() {
    void this.load();
  }

  private async load(): Promise<void> {
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
    try {
      this.groups.set(await this.library.listGroups());
    } catch {
      this.groups.set([]);
    }
    await this.refreshConversations();
  }

  private async refreshConversations(): Promise<void> {
    try {
      this.conversations.set(await this.chat.getConversations());
    } catch {
      this.conversations.set([]);
    }
  }

  newChat(): void {
    this.currentConversationId.set(null);
    this.messages.set([]);
    this.expanded.set(null);
  }

  onSelectConversation(value: string): void {
    if (!value) {
      this.newChat();
    } else {
      void this.loadConversation(Number(value));
    }
  }

  async loadConversation(id: number): Promise<void> {
    try {
      const conv = await this.chat.getConversation(id);
      this.currentConversationId.set(id);
      this.messages.set(
        conv.messages.map((m) => ({ role: m.role, text: m.content, sources: m.sources })),
      );
      this.expanded.set(null);
    } catch {
      /* ignore */
    }
  }

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
        group_id: this.selectedGroup() || null,
        conversation_id: this.currentConversationId(),
      };
      for await (const ev of this.chat.stream(request)) {
        if (ev.event === 'token') {
          this.draft.update((d) => d + (ev.data.content ?? ''));
        } else if (ev.event === 'done') {
          if (ev.data.conversation_id) this.currentConversationId.set(ev.data.conversation_id);
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
      void this.refreshConversations();
    }
  }

  private finish(message: Message): void {
    this.messages.update((list) => [...list, message]);
  }
}
