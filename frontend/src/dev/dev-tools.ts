/**
 * Nur im Dev-Build: Layout-Prüfer und – per ?mock=1 (bleibt bis ?mock=0 gemerkt) – Beispieldaten
 * statt Backend. Der Produktions-Build ersetzt diese Datei durch dev-tools.prod.ts
 * (fileReplacements in angular.json), Mock-Daten landen dort also gar nicht erst im Bundle.
 */
export async function installDevTools(): Promise<void> {
  (await import('./layout-check')).installLayoutCheck();
  const param = new URLSearchParams(location.search).get('mock');
  if (param === '1') sessionStorage.setItem('bookrag-mock', '1');
  if (param === '0') sessionStorage.removeItem('bookrag-mock');
  if (sessionStorage.getItem('bookrag-mock') === '1') (await import('./mock-api')).installMockApi();
}
