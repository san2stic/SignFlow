import { NavLink, useNavigate } from "react-router-dom";
import { Hand, LogOut } from "lucide-react";
import { useAuthStore } from "../../stores/authStore";
import { APP_NAV_ITEMS } from "./navItems";

export default function Sidebar() {
  const logout = useAuthStore((state) => state.logout);
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <aside className="hidden h-screen w-72 shrink-0 border-r border-slate-800/90 bg-slate-950/75 p-4 text-white backdrop-blur md:flex md:flex-col">
      <div className="rounded-card border border-slate-800/80 bg-slate-900/75 p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-secondary text-slate-950">
            <Hand className="w-5 h-5" />
          </div>
          <div>
            <p className="font-display text-lg font-semibold text-white">SignFlow</p>
            <p className="text-xs text-slate-400">Plateforme LSFB</p>
          </div>
        </div>
      </div>

      <nav className="mt-4 flex-1 space-y-1">
        {APP_NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-btn px-4 py-3 transition-all ${
                isActive
                  ? "bg-gradient-to-r from-primary/25 to-secondary/20 text-white ring-1 ring-primary/40"
                  : "text-slate-300 hover:bg-slate-800/70 hover:text-white"
              }`
            }
          >
            <item.icon className="h-5 w-5 shrink-0 text-current" />
            <span className="font-medium">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="space-y-3 border-t border-slate-800/80 pt-4">
        <div className="rounded-btn border border-slate-700/80 bg-slate-900/70 px-4 py-3">
          <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-400">Connecte en tant que</p>
          <p className="truncate text-sm font-semibold text-white">{user?.username}</p>
        </div>
        <button
          onClick={handleLogout}
          className="touch-btn flex w-full items-center gap-3 rounded-btn bg-slate-900/60 px-4 py-3 text-sm text-slate-300 hover:bg-red-500/20 hover:text-red-100"
        >
          <LogOut className="h-5 w-5" />
          <span className="font-medium">Se deconnecter</span>
        </button>
        <div className="pt-1 text-center text-xs text-slate-500">
          SignFlow v0.2.0
        </div>
      </div>
    </aside>
  );
}
