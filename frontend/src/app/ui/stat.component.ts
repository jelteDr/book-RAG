import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** Kennzahl: große Ziffern in SF Rounded, Größe folgt der Breite; die Einheit darf umbrechen. */
@Component({
  selector: 'ui-stat',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block min-w-0' },
  template: `
    <p class="text-xs text-ink-muted">{{ label() }}</p>
    <p class="font-display text-[clamp(1.375rem,5.5vw,2.25rem)] font-bold leading-none">
      <span class="whitespace-nowrap">{{ value() }}</span>
      @if (unit()) {
        <span class="font-sans text-sm font-normal text-ink-muted"> {{ unit() }}</span>
      }
    </p>
  `,
})
export class StatComponent {
  readonly label = input.required<string>();
  readonly value = input.required<string | number>();
  readonly unit = input('');
}
