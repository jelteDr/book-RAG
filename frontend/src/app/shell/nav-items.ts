import { ChartColumn, Cpu, Library, MessageSquare } from 'lucide-angular';

export interface NavItem {
  to: string;
  label: string;
  icon: typeof Cpu;
}

export const navItems: NavItem[] = [
  { to: '/chat', label: 'Chat', icon: MessageSquare },
  { to: '/groups', label: 'Dokumente', icon: Library },
  { to: '/models', label: 'Modelle', icon: Cpu },
  { to: '/dashboard', label: 'Dashboard', icon: ChartColumn },
];
