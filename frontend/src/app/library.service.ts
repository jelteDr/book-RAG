import { Injectable } from '@angular/core';

import { Book, Collection, Group, MetricsRow, ModelInfo, UploadResult } from './models';

/** REST-Zugriff auf Gruppen, Bücher, Modell-Metadaten und Metriken (via /api-Proxy). */
@Injectable({ providedIn: 'root' })
export class LibraryService {
  private async json<T>(url: string, init?: RequestInit): Promise<T> {
    const resp = await fetch(url, init);
    if (!resp.ok) throw new Error(`${url}: ${resp.status}`);
    return resp.json();
  }

  listGroups(): Promise<Group[]> {
    return this.json('/api/groups');
  }

  createGroup(body: Omit<Group, 'id' | 'n_books'>): Promise<Group> {
    return this.json('/api/groups', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  async deleteGroup(id: number): Promise<void> {
    await fetch(`/api/groups/${id}`, { method: 'DELETE' });
  }

  listCollections(): Promise<Collection[]> {
    return this.json('/api/collections');
  }

  createCollection(body: { slug: string; name: string; member_slugs: string[] }): Promise<Collection> {
    return this.json('/api/collections', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  async deleteCollection(id: number): Promise<void> {
    await fetch(`/api/collections/${id}`, { method: 'DELETE' });
  }

  listBooks(groupId: number): Promise<Book[]> {
    return this.json(`/api/groups/${groupId}/books`);
  }

  async deleteBook(id: number): Promise<void> {
    await fetch(`/api/books/${id}`, { method: 'DELETE' });
  }

  upload(
    groupId: number,
    file: File,
    opts: { commit: boolean; title?: string; author?: string },
  ): Promise<UploadResult> {
    const form = new FormData();
    form.append('file', file);
    form.append('commit', String(opts.commit));
    if (opts.title) form.append('title', opts.title);
    if (opts.author) form.append('author', opts.author);
    return this.json(`/api/groups/${groupId}/upload`, { method: 'POST', body: form });
  }

  modelsInfo(): Promise<ModelInfo[]> {
    return this.json('/api/models/info');
  }

  async patchModel(
    name: string,
    body: { cutoff_date?: string | null; parameter_count?: number | null },
  ): Promise<void> {
    await fetch(`/api/models/${name}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  metrics(): Promise<{ per_model: MetricsRow[] }> {
    return this.json('/api/metrics');
  }
}
