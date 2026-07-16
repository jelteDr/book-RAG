import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  OnDestroy,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';

import { ChatService } from '../chat.service';
import { LibraryService } from '../library.service';
import { Collection, Conversation, Group, Message, Source } from '../models';

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
export class ChatComponent implements OnDestroy {
  private readonly chat = inject(ChatService);
  private readonly library = inject(LibraryService);
  private readonly route = inject(ActivatedRoute);

  readonly models = signal<string[]>([]);
  readonly selectedModel = signal('');
  readonly groups = signal<Group[]>([]);
  readonly collections = signal<Collection[]>([]);
  readonly selectedGroup = signal('');
  readonly conversations = signal<Conversation[]>([]);
  readonly currentConversationId = signal<number | null>(null);
  readonly question = signal('');
  readonly messages = signal<Message[]>([]);
  readonly draft = signal('');
  readonly streaming = signal(false);
  readonly expanded = signal<Source | null>(null);

  /** Wechselnde Status-Sprüche, solange das Modell „denkt" (vor dem ersten Token). */
  private readonly thinkingWords = [
    'Denken',
    'Grübeln',
    'Sucht Quellen',
    'Liest Kontext',
    'Zeit für einen Schluck Kaffee',
    'Formuliert',
    'Tee aufsetzen lohnt sich',
    'Quellen werden gewogen',
    'Keks dazu?',
  ];
  private readonly thinkingIndex = signal(0);
  private thinkingTimer: ReturnType<typeof setInterval> | null = null;
  readonly thinkingWord = computed(
    () => this.thinkingWords[this.thinkingIndex() % this.thinkingWords.length],
  );

  /** Tickender Sekundenzähler über die gesamte Generierung (Denk-Phase + Streaming). */
  readonly elapsedSeconds = signal(0);
  private elapsedTimer: ReturnType<typeof setInterval> | null = null;

  /** Unterhaltungen nach Gruppe gebündelt (für die Sidebar). */
  readonly groupedConversations = computed(() => {
    const nameBySlug = new Map([
      ...this.groups().map((g) => [g.slug, g.name] as const),
      ...this.collections().map((c) => [c.slug, c.name] as const),
    ]);
    const buckets = new Map<string, { key: string; label: string; items: Conversation[] }>();
    for (const c of this.conversations()) {
      const key = c.group_id ?? '';
      if (!buckets.has(key)) {
        buckets.set(key, {
          key,
          label: key ? (nameBySlug.get(key) ?? key) : 'Ohne Gruppe',
          items: [],
        });
      }
      buckets.get(key)!.items.push(c);
    }
    return [...buckets.values()];
  });

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
    try {
      this.collections.set(await this.library.listCollections());
    } catch {
      this.collections.set([]);
    }
    // Aus „Chat starten" in der Dokumente-View: Gruppe/Sammelgruppe per ?group=<slug> vorauswählen.
    const preset = this.route.snapshot.queryParamMap.get('group');
    const known =
      this.groups().some((g) => g.slug === preset) ||
      this.collections().some((c) => c.slug === preset);
    if (preset && known) {
      this.selectedGroup.set(preset);
      this.newChat();
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

  /** Neuer Chat mit dieser Gruppe vorausgewählt (＋ neben dem Gruppennamen in der Sidebar). */
  newChatInGroup(slug: string): void {
    this.selectedGroup.set(slug);
    this.newChat();
  }

  /** Modell-Menü (Claude-Code-Stil: nur Name sichtbar, Klick öffnet die Auswahl). */
  readonly modelMenuOpen = signal(false);

  toggleModelMenu(): void {
    this.groupMenuOpen.set(false);
    this.modelMenuOpen.update((open) => !open);
  }

  selectModel(model: string): void {
    this.selectedModel.set(model);
    this.modelMenuOpen.set(false);
  }

  /** Gruppen-Menü im selben Stil; '' = alle Gruppen. */
  readonly groupMenuOpen = signal(false);
  readonly selectedGroupLabel = computed(() => {
    const slug = this.selectedGroup();
    if (!slug) return 'Alle Gruppen';
    return (
      this.groups().find((g) => g.slug === slug)?.name ??
      this.collections().find((c) => c.slug === slug)?.name ??
      slug
    );
  });

  toggleGroupMenu(): void {
    this.modelMenuOpen.set(false);
    this.groupMenuOpen.update((open) => !open);
  }

  selectGroup(slug: string): void {
    this.selectedGroup.set(slug);
    this.groupMenuOpen.set(false);
  }

  /** Eingeklappte Gruppen in der Sidebar (Key = group-slug, '' = „Ohne Gruppe"). */
  readonly collapsedGroups = signal<Set<string>>(new Set());

  toggleGroup(key: string): void {
    this.collapsedGroups.update((set) => {
      const next = new Set(set);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  isCollapsed(key: string): boolean {
    return this.collapsedGroups().has(key);
  }

  private startThinking(): void {
    this.thinkingIndex.set(0);
    this.stopThinking();
    // 2 s Takt, damit auch die längeren Sprüche lesbar bleiben.
    this.thinkingTimer = setInterval(() => this.thinkingIndex.update((i) => i + 1), 2000);
    this.elapsedSeconds.set(0);
    this.elapsedTimer = setInterval(() => this.elapsedSeconds.update((s) => s + 1), 1000);
  }

  private stopThinking(): void {
    if (this.thinkingTimer !== null) {
      clearInterval(this.thinkingTimer);
      this.thinkingTimer = null;
    }
    if (this.elapsedTimer !== null) {
      clearInterval(this.elapsedTimer);
      this.elapsedTimer = null;
    }
  }

  /** Millisekunden hübsch als Sekunden, z. B. 9358 -> "9,4 s". */
  fmtSecs(ms: number): string {
    return `${(ms / 1000).toFixed(1).replace('.', ',')} s`;
  }

  ngOnDestroy(): void {
    this.stopThinking();
  }

  async deleteConversation(conv: Conversation, event: Event): Promise<void> {
    event.stopPropagation(); // nicht gleichzeitig laden
    await this.chat.deleteConversation(conv.id);
    if (this.currentConversationId() === conv.id) this.newChat();
    await this.refreshConversations();
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
    this.startThinking();

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
      this.stopThinking();
      this.streaming.set(false);
      this.draft.set('');
      void this.refreshConversations();
    }
  }

  private finish(message: Message): void {
    this.messages.update((list) => [...list, message]);
  }
}
