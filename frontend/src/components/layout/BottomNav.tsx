import { NavLink } from "react-router-dom";

import { routeItems } from "../../routes";

export function BottomNav(): JSX.Element {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-slate-700 bg-surface/95 p-2 backdrop-blur md:hidden">
      <ul className="grid grid-cols-5 gap-1">
        {routeItems.map((route) => (
          <li key={route.path}>
            <NavLink
              to={route.path}
              end={route.path === "/translate"}
              className={({ isActive }) =>
                `flex touch-btn flex-col items-center justify-center gap-1 text-xs ${
                  isActive ? "bg-primary/20 text-primary" : "text-slate-300"
                }`
              }
            >
              <span aria-hidden>{route.icon}</span>
              <span>{route.label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
