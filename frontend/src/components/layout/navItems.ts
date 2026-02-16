import type { LucideIcon } from "lucide-react";
import { BookOpen, GraduationCap, Languages, LayoutDashboard, Settings, User } from "lucide-react";

export interface AppNavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  mobile?: boolean;
}

export const APP_NAV_ITEMS: AppNavItem[] = [
  { to: "/dashboard", icon: LayoutDashboard, label: "Dashboard", mobile: true },
  { to: "/translate", icon: Languages, label: "Traduire", mobile: true },
  { to: "/dictionary", icon: BookOpen, label: "Dictionnaire", mobile: true },
  { to: "/training", icon: GraduationCap, label: "Entrainement", mobile: true },
  { to: "/settings", icon: Settings, label: "Parametres", mobile: true },
  { to: "/profile", icon: User, label: "Profil", mobile: true }
];

export const MOBILE_NAV_ITEMS = APP_NAV_ITEMS.filter((item) => item.mobile !== false);
