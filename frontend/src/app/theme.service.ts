import { computed, Injectable, signal } from '@angular/core';

export type ThemeMode = 'system' | 'light' | 'dark';

// Derselbe Schlüssel wie im Bootstrap-Script in index.html.
const KEY = 'bookrag-theme';

/** Theme-Wahl (System/Hell/Dunkel); setzt data-theme am <html>, das die CSS-Tokens umschaltet. */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly media = window.matchMedia('(prefers-color-scheme: dark)');
  private readonly systemDark = signal(this.media.matches);

  readonly mode = signal<ThemeMode>(this.stored());
  readonly dark = computed(
    () => this.mode() === 'dark' || (this.mode() === 'system' && this.systemDark()),
  );

  constructor() {
    this.media.addEventListener('change', (event) => {
      this.systemDark.set(event.matches);
      this.apply();
    });
    this.apply();
  }

  setMode(mode: ThemeMode): void {
    this.mode.set(mode);
    this.apply();
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      // private Fenster o. Ä.: Auswahl gilt dann nur für diese Sitzung
    }
  }

  /** Schnellwechsel oben rechts: immer das Gegenteil des gerade sichtbaren Designs. */
  toggle(): void {
    this.setMode(this.dark() ? 'light' : 'dark');
  }

  private stored(): ThemeMode {
    try {
      const value = localStorage.getItem(KEY);
      return value === 'light' || value === 'dark' ? value : 'system';
    } catch {
      return 'system';
    }
  }

  private apply(): void {
    document.documentElement.dataset['theme'] = this.dark() ? 'dark' : 'light';
  }
}
