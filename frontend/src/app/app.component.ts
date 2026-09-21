import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  ElementRef,
  HostListener,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { LucideAngularModule, Menu, Moon, PanelLeftClose, PanelLeftOpen, Sun, X } from 'lucide-angular';
import { filter, map } from 'rxjs';

import { navItems } from './shell/nav-items';
import { SidebarComponent } from './shell/sidebar.component';
import { ThemeService } from './theme.service';

const COLLAPSED_KEY = 'bookrag-sidebar-collapsed';

interface RouteLayout {
  /** 'full' = Seite füllt die Höhe und scrollt selbst (Chat); 'page' = zentrierte Spalte. */
  layout?: 'full' | 'page';
  wide?: boolean;
}

/**
 * Rahmen der App: unter lg eine Topbar mit Hamburger, der die Sidebar als Drawer öffnet;
 * ab lg eine feste Sidebar, die sich auf eine Icon-Leiste einklappen lässt.
 */
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, SidebarComponent, LucideAngularModule],
  templateUrl: './app.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {
  readonly theme = inject(ThemeService);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  readonly icons = { Menu, Moon, PanelLeftClose, PanelLeftOpen, Sun, X };

  readonly drawerOpen = signal(false);
  readonly collapsed = signal(this.readCollapsed());

  private readonly menuButton = viewChild<ElementRef<HTMLButtonElement>>('menuButton');
  private readonly closeButton = viewChild<ElementRef<HTMLButtonElement>>('closeButton');

  private readonly url = toSignal(
    this.router.events.pipe(
      filter((event) => event instanceof NavigationEnd),
      map(() => this.router.url),
    ),
    { initialValue: this.router.url },
  );

  readonly pageTitle = computed(
    () => navItems.find((item) => this.url().startsWith(item.to))?.label ?? 'book-RAG',
  );

  readonly layout = computed<RouteLayout>(() => {
    this.url(); // nach jeder Navigation neu lesen
    let current = this.route.snapshot;
    while (current.firstChild) current = current.firstChild;
    return current.data as RouteLayout;
  });

  readonly contentClass = computed(() => {
    const { layout, wide } = this.layout();
    if (layout === 'full') return 'h-[calc(100dvh-3.5rem)]';
    return `mx-auto px-4 py-6 sm:px-6 lg:pb-10 lg:pt-2 ${wide ? 'max-w-6xl' : 'max-w-3xl'}`;
  });

  constructor() {
    // Fokus: beim Öffnen auf „Schließen“, danach zurück zum Hamburger (die Kopfzeile ist solange inert).
    let wasOpen = false;
    effect(() => {
      const open = this.drawerOpen();
      document.body.style.overflow = open ? 'hidden' : '';
      if (open) {
        this.closeButton()?.nativeElement.focus();
      } else if (wasOpen) {
        // erst nach dem Rendern: bis dahin ist die Kopfzeile noch inert und nimmt keinen Fokus an
        setTimeout(() => this.menuButton()?.nativeElement.focus());
      }
      wasOpen = open;
    });
  }

  @HostListener('document:keydown.escape')
  closeDrawer(): void {
    this.drawerOpen.set(false);
  }

  toggleCollapsed(): void {
    this.collapsed.update((value) => !value);
    try {
      localStorage.setItem(COLLAPSED_KEY, this.collapsed() ? '1' : '0');
    } catch {
      // ohne Speicher gilt die Wahl nur für diese Sitzung
    }
  }

  private readCollapsed(): boolean {
    try {
      return localStorage.getItem(COLLAPSED_KEY) === '1';
    } catch {
      return false;
    }
  }
}
