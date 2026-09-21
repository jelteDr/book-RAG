import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** Seitentitel als „Large Title“ (genau eine h1 pro Seite); Aktionen per `<… actions>`-Slot rechts. */
@Component({
  selector: 'ui-page-header',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
  template: `
    <header class="flex flex-wrap items-end justify-between gap-x-3 gap-y-2">
      <div class="min-w-0">
        @if (eyebrow()) {
          <p class="text-[13px] font-semibold text-ink-muted">{{ eyebrow() }}</p>
        }
        <h1 class="break-words text-[28px] font-bold leading-tight tracking-tight sm:text-[34px]">
          {{ title() }}
        </h1>
      </div>
      <div class="flex shrink-0 items-center gap-2 empty:hidden"><ng-content select="[actions]" /></div>
    </header>
  `,
})
export class PageHeaderComponent {
  readonly title = input.required<string>();
  readonly eyebrow = input('');
}
