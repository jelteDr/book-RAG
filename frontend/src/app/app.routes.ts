import { Routes } from '@angular/router';

import { ChatComponent } from './chat/chat.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { GroupsComponent } from './groups/groups.component';
import { ModelsComponent } from './models/models.component';

// data.layout 'full' = Seite füllt die Höhe (Chat); data.wide = breite Inhaltsspalte (AppComponent).
export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'chat' },
  { path: 'chat', component: ChatComponent, data: { layout: 'full' } },
  { path: 'groups', component: GroupsComponent },
  { path: 'models', component: ModelsComponent, data: { wide: true } },
  { path: 'dashboard', component: DashboardComponent, data: { wide: true } },
  { path: '**', redirectTo: 'chat' },
];
