import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter } from '@angular/router';

import { AppComponent } from './app/app.component';
import { routes } from './app/app.routes';
import { installDevTools } from './dev/dev-tools';

// Dev-Werkzeuge (Mock-API muss fetch ersetzen, bevor die App lädt); im Produktions-Build ein No-op.
installDevTools()
  .then(() => bootstrapApplication(AppComponent, { providers: [provideRouter(routes)] }))
  .catch((err) => console.error(err));
