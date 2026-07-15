import { Routes } from '@angular/router';

import { ChatComponent } from './chat/chat.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { GroupsComponent } from './groups/groups.component';
import { ModelsComponent } from './models/models.component';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'chat' },
  { path: 'chat', component: ChatComponent },
  { path: 'groups', component: GroupsComponent },
  { path: 'models', component: ModelsComponent },
  { path: 'dashboard', component: DashboardComponent },
  { path: '**', redirectTo: 'chat' },
];
