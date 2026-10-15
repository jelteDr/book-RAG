import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import {
  FileText,
  FolderOpen,
  Layers,
  LucideAngularModule,
  MessageSquare,
  Pencil,
  Plus,
  Trash2,
} from 'lucide-angular';

import { LibraryService } from '../library.service';
import { Book, Collection, Group, UploadResult } from '../models';
import { BannerComponent } from '../ui/banner.component';
import { EmptyStateComponent } from '../ui/empty-state.component';
import { PageHeaderComponent } from '../ui/page-header.component';
import { SectionComponent } from '../ui/section.component';

@Component({
  selector: 'app-groups',
  standalone: true,
  imports: [
    FormsModule,
    LucideAngularModule,
    BannerComponent,
    EmptyStateComponent,
    PageHeaderComponent,
    SectionComponent,
  ],
  host: { class: 'stagger block space-y-7' },
  templateUrl: './groups.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GroupsComponent {
  private readonly library = inject(LibraryService);
  private readonly router = inject(Router);

  readonly groups = signal<Group[]>([]);
  readonly error = signal('');
  readonly loading = signal(true);

  readonly icons = { FileText, FolderOpen, Layers, MessageSquare, Pencil, Plus, Trash2 };

  // Titel/Autor eines Buchs inline bearbeiten (eine Zeile zur Zeit).
  readonly editingBookId = signal<number | null>(null);
  readonly editTitle = signal('');
  readonly editAuthor = signal('');
  readonly editError = signal('');
  readonly editSaving = signal(false);

  /** Löschen fragt einmal in der Zeile nach; Schlüssel z. B. "group:3", "book:12", "collection:1". */
  readonly confirmDelete = signal<string | null>(null);

  // Formular "neue Gruppe" (Slug erzeugt das Backend aus dem Namen)
  readonly name = signal('');
  readonly kind = signal('');

  // Upload-Panel
  readonly uploadGroupId = signal<number | null>(null);
  readonly file = signal<File | null>(null);
  readonly uploadTitle = signal('');
  readonly uploadAuthor = signal('');
  readonly report = signal<UploadResult | null>(null);
  readonly busy = signal(false);

  // Bücher je Gruppe (immer sichtbar = Dokumentenverwaltung)
  readonly booksByGroup = signal<Record<number, Book[]>>({});

  // Sammelgruppen (bündeln Gruppen ohne Re-Upload; Slug erzeugt das Backend)
  readonly collections = signal<Collection[]>([]);
  readonly colName = signal('');
  readonly colMembers = signal<Set<string>>(new Set());

  constructor() {
    void this.loadGroups();
  }

  booksOf(groupId: number): Book[] {
    return this.booksByGroup()[groupId] ?? [];
  }

  private async loadGroups(): Promise<void> {
    try {
      const groups = await this.library.listGroups();
      this.groups.set(groups);
      const entries = await Promise.all(
        groups.map(async (g) => [g.id, await this.library.listBooks(g.id)] as const),
      );
      this.booksByGroup.set(Object.fromEntries(entries));
      this.collections.set(await this.library.listCollections());
    } catch (e) {
      this.error.set(`Gruppen laden fehlgeschlagen: ${e}`);
    } finally {
      this.loading.set(false);
    }
  }

  toggleMember(slug: string): void {
    this.colMembers.update((set) => {
      const next = new Set(set);
      if (next.has(slug)) {
        next.delete(slug);
      } else {
        next.add(slug);
      }
      return next;
    });
  }

  async createCollection(): Promise<void> {
    const name = this.colName().trim();
    const members = [...this.colMembers()];
    if (!name || !members.length) return;
    try {
      await this.library.createCollection({ name, member_slugs: members });
      this.colName.set('');
      this.colMembers.set(new Set());
      this.collections.set(await this.library.listCollections());
    } catch (e) {
      this.error.set(`Sammelgruppe anlegen fehlgeschlagen: ${e}`);
    }
  }

  async deleteCollection(collection: Collection): Promise<void> {
    this.confirmDelete.set(null);
    await this.library.deleteCollection(collection.id);
    this.collections.set(await this.library.listCollections());
  }

  /** Chat mit dieser Sammelgruppe vorausgewählt öffnen (Slug-Namensraum ist geteilt). */
  openCollectionChat(collection: Collection): void {
    void this.router.navigate(['/chat'], { queryParams: { group: collection.slug } });
  }

  async createGroup(): Promise<void> {
    const name = this.name().trim();
    if (!name) return;
    try {
      await this.library.createGroup({
        name,
        kind: this.kind().trim() || null,
        description: null,
      });
      this.name.set('');
      this.kind.set('');
      await this.loadGroups();
    } catch (e) {
      this.error.set(`Anlegen fehlgeschlagen: ${e}`);
    }
  }

  /** Öffnet den Chat mit dieser Gruppe bereits vorausgewählt (frischer Chat). */
  openChat(group: Group): void {
    void this.router.navigate(['/chat'], { queryParams: { group: group.slug } });
  }

  async deleteGroup(group: Group): Promise<void> {
    this.confirmDelete.set(null);
    await this.library.deleteGroup(group.id);
    await this.loadGroups();
  }

  startEdit(book: Book): void {
    this.confirmDelete.set(null);
    this.editingBookId.set(book.id);
    this.editTitle.set(book.title ?? '');
    this.editAuthor.set(book.author ?? '');
    this.editError.set('');
  }

  cancelEdit(): void {
    this.editingBookId.set(null);
    this.editError.set('');
  }

  async saveEdit(book: Book, groupId: number): Promise<void> {
    const title = this.editTitle().trim();
    if (!title) {
      this.editError.set('Titel darf nicht leer sein.');
      return;
    }
    this.editSaving.set(true);
    try {
      const updated = await this.library.patchBook(book.id, { title, author: this.editAuthor().trim() });
      this.booksByGroup.update((m) => ({
        ...m,
        [groupId]: (m[groupId] ?? []).map((b) => (b.id === book.id ? updated : b)),
      }));
      this.editingBookId.set(null);
    } catch (e) {
      this.editError.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.editSaving.set(false);
    }
  }

  async deleteBook(book: Book): Promise<void> {
    this.confirmDelete.set(null);
    await this.library.deleteBook(book.id);
    await this.loadGroups();
  }

  openUpload(group: Group): void {
    this.uploadGroupId.set(group.id);
    this.file.set(null);
    this.uploadTitle.set('');
    this.uploadAuthor.set('');
    this.report.set(null);
  }

  private uploadOpts(commit: boolean): { commit: boolean; title?: string; author?: string } {
    return {
      commit,
      title: this.uploadTitle().trim() || undefined,
      author: this.uploadAuthor().trim() || undefined,
    };
  }

  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.file.set(input.files?.[0] ?? null);
    this.report.set(null);
  }

  async dryRun(group: Group): Promise<void> {
    const f = this.file();
    if (!f) return;
    this.busy.set(true);
    try {
      const res = await this.library.upload(group.id, f, this.uploadOpts(false));
      this.report.set(res);
      // Erkannten/Fallback-Titel vorbefüllen, damit man ihn sieht und korrigieren kann.
      if (!this.uploadTitle() && res.title) this.uploadTitle.set(res.title);
      if (!this.uploadAuthor() && res.author) this.uploadAuthor.set(res.author);
    } catch (e) {
      this.error.set(`Dry-Run fehlgeschlagen: ${e}`);
    } finally {
      this.busy.set(false);
    }
  }

  async commit(group: Group): Promise<void> {
    const f = this.file();
    if (!f) return;
    this.busy.set(true);
    try {
      const res = await this.library.upload(group.id, f, this.uploadOpts(true));
      if (!res.committed) {
        // Validierung fehlgeschlagen -> Report mit Fehlern zeigen, Panel offen lassen.
        this.report.set(res);
        return;
      }
      this.uploadGroupId.set(null);
      this.file.set(null);
      this.report.set(null);
      await this.loadGroups();
    } catch (e) {
      this.error.set(`Upload fehlgeschlagen: ${e}`);
    } finally {
      this.busy.set(false);
    }
  }
}
