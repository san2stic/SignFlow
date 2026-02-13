import { Outlet } from "react-router-dom";

import { BottomNav } from "./BottomNav";
import { Sidebar } from "./Sidebar";

export function PageShell(): JSX.Element {
  return (
    <div className="mx-auto flex min-h-screen max-w-7xl text-text">
      <aside className="hidden w-64 border-r border-slate-800/80 p-4 md:block">
        <Sidebar />
      </aside>
      <main className="flex-1 p-3 pb-24 md:p-6 md:pb-6">
        <Outlet />
      </main>
      <BottomNav />
    </div>
  );
}
