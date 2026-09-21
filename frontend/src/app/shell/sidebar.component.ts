import { booleanAttribute, ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { BookOpen, LucideAngularModule } from 'lucide-angular';

import { navItems } from './nav-items';

/** Reine Navigation – der Hell/Dunkel-Wechsel liegt oben rechts in der Kopfzeile. */
@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block h-full' },
  template: `
    <div class="flex h-full flex-col bg-bar backdrop-blur-xl">
      <div class="flex h-16 shrink-0 items-center gap-2.5" [class]="collapsed() ? 'justify-center' : 'px-5'">
        <span
          class="grid size-8 place-items-center rounded-[9px] bg-gradient-to-b from-[#34c759] to-[#1c7331] text-white shadow-thumb"
        >
          <lucide-icon [img]="BookOpen" [size]="18" aria-hidden="true" />
        </span>
        @if (!collapsed()) {
          <span class="text-[17px] font-semibold tracking-tight">book-RAG</span>
        }
      </div>

      <nav aria-label="Hauptnavigation" class="flex-1 overflow-y-auto p-3">
        <ul class="space-y-0.5">
          @for (item of items; track item.to) {
            <li>
              <a
                [routerLink]="item.to"
                routerLinkActive="!bg-accent font-semibold !text-accent-fg"
                ariaCurrentWhenActive="page"
                [attr.title]="collapsed() ? item.label : null"
                [attr.aria-label]="collapsed() ? item.label : null"
                class="flex h-10 items-center gap-3 rounded-[10px] text-[15px] text-ink transition-colors hover:bg-surface-muted"
                [class]="collapsed() ? 'justify-center px-0' : 'px-3'"
                (click)="navigate.emit()"
              >
                <lucide-icon [img]="item.icon" [size]="18" [strokeWidth]="2.2" aria-hidden="true" class="shrink-0" />
                @if (!collapsed()) {
                  {{ item.label }}
                }
              </a>
            </li>
          }
        </ul>
      </nav>
    </div>
  `,
})
export class SidebarComponent {
  /** nur Icons (eingeklappte Desktop-Sidebar) */
  readonly collapsed = input(false, { transform: booleanAttribute });
  readonly navigate = output<void>();

  readonly items = navItems;
  readonly BookOpen = BookOpen;
}
