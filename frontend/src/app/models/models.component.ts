import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Check, LoaderCircle, LucideAngularModule, RefreshCw } from 'lucide-angular';

import { LibraryService } from '../library.service';
import { MetricsRow, ModelInfo } from '../models';
import { BannerComponent } from '../ui/banner.component';
import { PageHeaderComponent } from '../ui/page-header.component';
import { SectionComponent } from '../ui/section.component';

@Component({
  selector: 'app-models',
  standalone: true,
  imports: [FormsModule, LucideAngularModule, BannerComponent, PageHeaderComponent, SectionComponent],
  host: { class: 'stagger block space-y-7' },
  templateUrl: './models.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ModelsComponent {
  private readonly library = inject(LibraryService);

  readonly infos = signal<ModelInfo[]>([]);
  readonly metrics = signal<Record<string, MetricsRow>>({});
  readonly cutoffDraft = signal<Record<string, string>>({});
  readonly busy = signal(false);
  readonly error = signal('');
  readonly loading = signal(true);

  readonly icons = { Check, LoaderCircle, RefreshCw };

  readonly rows = computed(() =>
    this.infos().map((m) => ({ info: m, metric: this.metrics()[m.name] })),
  );

  constructor() {
    void this.load();
  }

  private async load(): Promise<void> {
    try {
      const [infos, metrics] = await Promise.all([
        this.library.modelsInfo(),
        this.library.metrics(),
      ]);
      this.infos.set(infos);
      this.metrics.set(Object.fromEntries(metrics.per_model.map((r) => [r.model, r])));
      this.cutoffDraft.set(
        Object.fromEntries(infos.map((m) => [m.name, m.cutoff_date ?? ''])),
      );
    } catch (e) {
      this.error.set(`Laden fehlgeschlagen: ${e}`);
    } finally {
      this.loading.set(false);
    }
  }

  setCutoff(name: string, value: string): void {
    this.cutoffDraft.update((d) => ({ ...d, [name]: value }));
  }

  async saveCutoff(name: string): Promise<void> {
    try {
      await this.library.patchModel(name, { cutoff_date: this.cutoffDraft()[name] || null });
      await this.load();
    } catch (e) {
      this.error.set(`Speichern fehlgeschlagen: ${e}`);
    }
  }

  async sync(): Promise<void> {
    this.busy.set(true);
    try {
      await fetch('/api/models/sync', { method: 'POST' });
      await this.load();
    } finally {
      this.busy.set(false);
    }
  }
}
