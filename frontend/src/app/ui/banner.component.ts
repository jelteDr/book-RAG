import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { CircleAlert, Info, LucideAngularModule, TriangleAlert } from 'lucide-angular';

const tones = {
  info: { box: 'bg-info-soft text-info', icon: Info },
  warn: { box: 'bg-warn-soft text-warn', icon: TriangleAlert },
  danger: { box: 'bg-danger-soft text-danger', icon: CircleAlert },
};

/** Inline-Hinweis mit Icon + Text (nie Farbe allein). tone="danger" wird als role="alert" vorgelesen. */
@Component({
  selector: 'ui-banner',
  standalone: true,
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
  template: `
    <div
      [attr.role]="tone() === 'danger' ? 'alert' : null"
      class="flex items-start gap-2 rounded-xl px-3.5 py-2.5 text-sm"
      [class]="style().box"
    >
      <lucide-icon [img]="style().icon" [size]="16" aria-hidden="true" class="mt-0.5 shrink-0" />
      <div class="min-w-0 break-words text-ink"><ng-content /></div>
    </div>
  `,
})
export class BannerComponent {
  readonly tone = input<keyof typeof tones>('info');
  readonly style = computed(() => tones[this.tone()]);
}
