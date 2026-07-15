import { Injectable, signal } from '@angular/core';

type Theme = 'light' | 'dark';

/** Verwaltet Light/Dark-Mode: Signal + localStorage + OS-Präferenz, toggelt die `dark`-Klasse. */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly theme = signal<Theme>(this.initial());

  constructor() {
    this.apply(this.theme());
  }

  toggle(): void {
    const next: Theme = this.theme() === 'dark' ? 'light' : 'dark';
    this.theme.set(next);
    this.apply(next);
    localStorage.setItem('theme', next);
  }

  private initial(): Theme {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  private apply(theme: Theme): void {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }
}
