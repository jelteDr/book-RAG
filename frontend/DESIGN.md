# book-RAG – Designsystem (Frontend)

Angular-Port von Jeltes Haus-Design („Apple-Stil“, Ursprung: GymTracker/React). Der Techstack bleibt
Angular 18 (Standalone, Signals, OnPush) + Tailwind 3.4; übernommen sind Tokens, Bausteine und Regeln.

## Richtung

1. **Grauer Grund, weiße Karten** (16 px Radius, ohne Rahmen/Schatten); Zeilen trennt eine Haarlinie
   (`divide-line`). Abschnittstitel stehen klein, grau, in Versalien *über* der Karte (`ui-section`).
2. **Systemschrift** (`font-sans` = San Francisco); Kennzahlen in `font-display` (SF Rounded, fett).
   Seitentitel sind „Large Titles“ (28/34 px, genau eine `h1` pro Seite).
3. **Bedienelemente:** gefüllt (primär), getönt (sekundär), Text (ghost/danger); 12 px Radius, 44 px hoch.
4. **Durchscheinende Leisten** (`bg-bar backdrop-blur-xl`) für Sidebar, Topbar, Chat-Eingabe; Menüs `.menu`.
5. **Ein Akzent:** Grün `#1c7331` (hell) / `#30d158` (dunkel), Kontraste ≥ 4,5:1.
   **Diagramme nutzen Datenfarben** (`bg-chart-1` Blau, `bg-chart-2` Orange), nie den Akzent.
6. **Dark Mode:** echtes Schwarz, Karten `#1c1c1e`; folgt dem System, umschaltbar (`ThemeService`,
   `data-theme` am `<html>`, Bootstrap-Script in `index.html`, Schlüssel `bookrag-theme`).

## Tokens (`src/styles.css` → `tailwind.config.js`)

Templates benutzen nur semantische Klassen – nie `slate-*`, `bg-white` oder Hex-Werte.

| Token | Klasse (Beispiel) | Zweck |
|---|---|---|
| `canvas` / `surface` / `surface-muted` | `bg-surface` | Seitengrund / Karten, Felder / Chips, Hover, Sekundär-Buttons |
| `line`, `line-strong`, `line-soft` | `divide-line` | Haarlinien / Feldrahmen (≥ 3:1) / Leistenränder |
| `ink`, `ink-muted` | `text-ink-muted` | Text / Sekundärtext |
| `accent` (+ `-hover`, `-fg`, `-soft`) | `bg-accent text-accent-fg` | Akzent; `accent-soft` für Icon-Kacheln (`.tile`) |
| `danger`, `info`, `warn` (+ `-soft`) | `bg-warn-soft text-warn` | Meldungen, ungestützte Zitate |
| `bar`, `overlay` | `bg-bar` | Leisten, Drawer-Overlay |
| `chart-1`, `chart-2` | `bg-chart-1` | Diagramm-Serien |

Radien: `rounded-sm` 10 px (Felder), `rounded-xl` 12 px (Buttons, Banner), `rounded-md` 16 px (Karten).
Schatten: `shadow-thumb`, `shadow-pop`. Tailwind v3 kann bei `var()`-Farben keine `/25`-Deckkraft –
dafür gibt es eigene Tokens (`--accent-ring`, `--line-soft`, `--menu`).

## Bausteine

**Klassen** (`@layer components` in `styles.css`) – für native Elemente, idiomatischer als Wrapper-Komponenten:
`.btn` + `.btn-primary|secondary|ghost|danger` (+ `.btn-sm`), `.icon-btn` (44 px, rund, **braucht `aria-label`**),
`.control` (Felder, Fokus-Ring, `aria-invalid`), `.field-label`, `.menu` / `.menu-item` / `.menu-title`, `.tile`.

**Komponenten** (`src/app/ui/`): `ui-page-header` (Slot `[actions]`), `ui-section` (`title`, `card`, `padded`,
Slot `[aside]`), `ui-stat`, `ui-empty-state`, `ui-banner` (`tone`; danger = `role="alert"`).

**Shell** (`src/app/shell/`, `app.component.*`): ab `lg` feste Sidebar (240 px, einklappbar auf 64 px,
Schlüssel `bookrag-sidebar-collapsed`); darunter Topbar + Drawer (`role="dialog"`, Esc/Overlay/Navigation
schließt, Fokus-Rückgabe, Rest `inert`). Route-Daten: `layout: 'full'` (Chat füllt die Höhe), `wide: true`
(breite Spalte). Neue Seite: Eintrag in `nav-items.ts` + Route; Host-Klasse `stagger block space-y-7`.

Icons: `lucide-angular`, Einzelimporte (`<lucide-icon [img]="icons.Trash2" [size]="18" aria-hidden="true" />`).

## Regeln

- **Löschen fragt in der Zeile nach** („Löschen“/„Nein“ ersetzen die Aktionen) – kein `confirm()`, kein Modal.
- **Zustände:** Laden = ruhiger Text „Lädt …“; leer = `ui-empty-state`; Fehler = `ui-banner tone="danger"`
  mit Handlungsanweisung. Status nie nur über Farbe (ungestützte Zitate: Warnfarbe **und** Icon).
- **Responsiv:** schrumpfende Grid/Flex-Kinder `min-w-0`, Spalten `minmax(0,1fr)`; freie Texte `truncate` +
  `max-w-full`; Kennzahlen mit `clamp()`. Tabellen scrollen *in* der Karte (`overflow-x-auto`).
- **`backdrop-filter` nicht auf Container mit `fixed`-Kindern** (begrenzt deren Bezugsrahmen) – Blur auf
  eigene Ebene legen (siehe Chat-Eingabe).
- Texte: Deutsch, du-Form, knapp; Buttons sind Verben.

## Prüfen (nur Dev)

`npm start`, dann `http://localhost:4200/?mock=1`: `src/dev/mock-api.ts` beantwortet `/api/*` mit absichtlich
sperrigen Beispieldaten (lange Namen, große Zahlen, ungestütztes Zitat, Validierungsfehler); `?mock=0` beendet
den Modus. In der Konsole listet `__layoutCheck()` alles, was aus Fenster oder Karte ragt. Jede geänderte Seite
bei ~345, ~820 und ≥ 1280 px, hell und dunkel ansehen. Der Produktions-Build ersetzt `src/dev/dev-tools.ts`
durch ein No-op (`fileReplacements`) – Mock-Daten landen nicht im Bundle; im Backend gibt es keinen Schalter.
