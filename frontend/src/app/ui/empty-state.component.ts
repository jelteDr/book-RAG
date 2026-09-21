import { ChangeDetectionStrategy, Component } from '@angular/core';

/** Leerer Zustand: ein Satz, ggf. mit Link zur Lösung. Nicht für „lädt“ benutzen. */
@Component({
  selector: 'ui-empty-state',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block rounded-md bg-surface px-4 py-8 text-center text-sm text-ink-muted' },
  template: `<ng-content />`,
})
export class EmptyStateComponent {}
