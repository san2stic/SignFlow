import { NavLink } from "react-router-dom";
import { MOBILE_NAV_ITEMS } from "./navItems";

export function BottomNav(): JSX.Element {
  const mobileItems = MOBILE_NAV_ITEMS;

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-slate-700/80 bg-slate-950/90 px-2 pb-[max(0.6rem,env(safe-area-inset-bottom))] pt-2 backdrop-blur md:hidden">
      <ul className="grid gap-1" style={{ gridTemplateColumns: `repeat(${mobileItems.length}, minmax(0, 1fr))` }}>
        {mobileItems.map((route) => (
          <li key={route.to}>
            <NavLink
              to={route.to}
              end={route.to === "/dashboard"}
              className={({ isActive }) =>
                `flex min-h-14 flex-col items-center justify-center gap-1 rounded-btn px-1 text-[11px] font-medium transition ${
                  isActive
                    ? "bg-gradient-to-r from-primary/25 to-secondary/25 text-white ring-1 ring-primary/40"
                    : "text-slate-300 hover:bg-slate-800/75"
                }`
              }
            >
              <route.icon aria-hidden className="h-4 w-4" />
              <span className="truncate">{route.label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
