import { NavLink } from "react-router-dom";

import { routeItems } from "../../routes";

export function Sidebar(): JSX.Element {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-heading text-2xl font-bold text-text">SignFlow</h1>
        <p className="text-sm text-slate-400">Live sign translation</p>
      </header>

      <nav>
        <ul className="space-y-2">
          {routeItems.map((route) => (
            <li key={route.path}>
              <NavLink
                to={route.path}
                end={route.path === "/translate"}
                className={({ isActive }) =>
                  `flex items-center gap-2 rounded-btn px-3 py-3 text-sm ${
                    isActive ? "bg-primary/20 text-primary" : "text-slate-200 hover:bg-slate-700/40"
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
    </div>
  );
}
