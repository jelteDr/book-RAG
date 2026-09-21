/**
 * Dev-Werkzeug: __layoutCheck() in der Konsole listet alles, was aus dem Fenster oder aus seiner
 * Karte ragt. Mit sperrigen Daten prüfen (?mock=1) – „kein horizontales Scrollen“ reicht nicht.
 */
export function installLayoutCheck(): void {
  (window as unknown as { __layoutCheck: () => unknown }).__layoutCheck = () => {
    const width = document.documentElement.clientWidth;
    const problems: { element: string; problem: string }[] = [];
    const describe = (el: Element) =>
      `${el.tagName.toLowerCase()}.${String(el.className).split(' ').slice(0, 3).join('.')} „${(el.textContent ?? '').trim().slice(0, 40)}“`;
    for (const el of Array.from(document.body.querySelectorAll('*'))) {
      const rect = el.getBoundingClientRect();
      if (!rect.width || !rect.height) continue;
      // Absichtlich scrollbare Bereiche (Tabellen) und deren Inhalt auslassen
      if (el.closest('.overflow-x-auto')) continue;
      if (rect.right > width + 1 || rect.left < -1) problems.push({ element: describe(el), problem: 'ragt aus dem Fenster' });
      const card = el.parentElement?.closest('.rounded-md');
      if (card) {
        const box = card.getBoundingClientRect();
        if (rect.right > box.right + 1 || rect.left < box.left - 1) problems.push({ element: describe(el), problem: 'ragt aus der Karte' });
      }
    }
    console.table(problems);
    return problems;
  };
}
