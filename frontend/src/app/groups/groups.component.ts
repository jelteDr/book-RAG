import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { LibraryService } from '../library.service';
import { Book, Group, UploadResult } from '../models';

@Component({
  selector: 'app-groups',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './groups.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GroupsComponent {
  private readonly library = inject(LibraryService);

  readonly groups = signal<Group[]>([]);
  readonly error = signal('');

  // Formular "neue Gruppe"
  readonly slug = signal('');
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
    } catch (e) {
      this.error.set(`Gruppen laden fehlgeschlagen: ${e}`);
    }
  }

  async createGroup(): Promise<void> {
    const slug = this.slug().trim();
    const name = this.name().trim();
    if (!slug || !name) return;
    try {
      await this.library.createGroup({
        slug,
        name,
        kind: this.kind().trim() || null,
        description: null,
      });
      this.slug.set('');
      this.name.set('');
      this.kind.set('');
      await this.loadGroups();
    } catch (e) {
      this.error.set(`Anlegen fehlgeschlagen: ${e}`);
    }
  }

  async deleteGroup(group: Group): Promise<void> {
    await this.library.deleteGroup(group.id);
    await this.loadGroups();
  }

  async deleteBook(book: Book): Promise<void> {
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
      this.report.set(await this.library.upload(group.id, f, this.uploadOpts(false)));
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
      await this.library.upload(group.id, f, this.uploadOpts(true));
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
