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
  readonly report = signal<UploadResult | null>(null);
  readonly busy = signal(false);

  // Bücher-Ansicht
  readonly expandedGroupId = signal<number | null>(null);
  readonly books = signal<Book[]>([]);

  constructor() {
    void this.loadGroups();
  }

  private async loadGroups(): Promise<void> {
    try {
      this.groups.set(await this.library.listGroups());
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
    if (this.expandedGroupId() === group.id) this.expandedGroupId.set(null);
    await this.loadGroups();
  }

  async toggleBooks(group: Group): Promise<void> {
    if (this.expandedGroupId() === group.id) {
      this.expandedGroupId.set(null);
      return;
    }
    this.books.set(await this.library.listBooks(group.id));
    this.expandedGroupId.set(group.id);
  }

  async deleteBook(book: Book, group: Group): Promise<void> {
    await this.library.deleteBook(book.id);
    this.books.set(await this.library.listBooks(group.id));
    await this.loadGroups();
  }

  openUpload(group: Group): void {
    this.uploadGroupId.set(group.id);
    this.file.set(null);
    this.report.set(null);
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
      this.report.set(await this.library.upload(group.id, f, { commit: false }));
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
      await this.library.upload(group.id, f, { commit: true });
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
