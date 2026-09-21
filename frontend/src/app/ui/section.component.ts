import { booleanAttribute, ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * Abschnitt im Apple-Stil („inset grouped“): kleine graue Überschrift, darunter eine weiße Karte.
 * Listen in der Karte trennen ihre Zeilen mit divide-line ([padded]="false").
 * [card]="false" lässt die Karte weg, wenn der Inhalt selbst Karten oder Chips mitbringt.
 */
@Component({
  selector: 'ui-section',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block min-w-0' },
  template: `
    <section class="min-w-0">
      <div class="mb-2 flex min-h-8 flex-wrap items-center justify-between gap-x-3 gap-y-2 px-4">
        <h2 class="text-[13px] font-medium uppercase tracking-wide text-ink-muted">{{ title() }}</h2>
        <ng-content select="[aside]" />
      </div>
      <div [class]="card() ? (padded() ? 'min-w-0 rounded-md bg-surface p-4 sm:p-5' : 'min-w-0 rounded-md bg-surface px-4') : 'min-w-0'">
        <ng-content />
      </div>
    </section>
  `,
})
export class SectionComponent {
  readonly title = input.required<string>();
  readonly card = input(true, { transform: booleanAttribute });
  readonly padded = input(true, { transform: booleanAttribute });
}
