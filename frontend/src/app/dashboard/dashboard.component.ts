import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { LibraryService } from '../library.service';
import { MetricsRow, ModelInfo } from '../models';
import { BannerComponent } from '../ui/banner.component';
import { EmptyStateComponent } from '../ui/empty-state.component';
import { PageHeaderComponent } from '../ui/page-header.component';
import { SectionComponent } from '../ui/section.component';
import { StatComponent } from '../ui/stat.component';

interface Card {
  info: ModelInfo;
  metric?: MetricsRow;
}

/** Ein Balken im Vergleichs-Chart: Wert (null = keine Telemetrie) + relative Höhe (0–100 %). */
interface Bar {
  model: string;
  value: number | null;
  pct: number;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [BannerComponent, EmptyStateComponent, PageHeaderComponent, SectionComponent, StatComponent],
  host: { class: 'stagger block space-y-7' },
  templateUrl: './dashboard.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardComponent {
  private readonly library = inject(LibraryService);

  readonly infos = signal<ModelInfo[]>([]);
  readonly metrics = signal<Record<string, MetricsRow>>({});
  readonly error = signal('');
  readonly loading = signal(true);

  readonly cards = computed<Card[]>(() =>
    this.infos().map((info) => ({ info, metric: this.metrics()[info.name] })),
  );
  readonly totalQueries = computed(() =>
    Object.values(this.metrics()).reduce((sum, m) => sum + (m.queries ?? 0), 0),
  );

  /** TPS-Vergleich (mehr = schneller). */
  readonly tpsBars = computed(() => this.bars((m) => m.avg_tps));
  /** TTFT-Vergleich in ms (weniger = schneller). */
  readonly ttftBars = computed(() => this.bars((m) => m.avg_ttft_ms));

  /**
   * Normierte Balken über ALLE Modelle (gleiche Reihenfolge wie die Karten,
   * damit ein Modell in beiden Charts an derselben x-Position steht).
   * Modelle ohne Telemetrie: value=null → grauer Stummel-Balken.
   */
  private bars(pick: (m: MetricsRow) => number | null): Bar[] {
    const rows = this.infos().map((info) => {
      const raw = this.metrics()[info.name] ? pick(this.metrics()[info.name]) : null;
      return { model: info.name, value: raw != null && raw > 0 ? raw : null };
    });
    const max = rows.reduce((mx, r) => Math.max(mx, r.value ?? 0), 0);
    return rows.map((r) => ({ ...r, pct: r.value && max ? (r.value / max) * 100 : 0 }));
  }

  /** Millisekunden als Sekunden mit Dezimalkomma (9358 -> "9,4"); große ms-Zahlen sprengen schmale Spalten. */
  secs(ms: number | null | undefined): string {
    return ms == null ? '—' : (ms / 1000).toFixed(1).replace('.', ',');
  }

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
    } catch (e) {
      this.error.set(`Laden fehlgeschlagen: ${e}`);
    } finally {
      this.loading.set(false);
    }
  }
}
