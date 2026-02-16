import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import { BottomNav } from "./BottomNav";

export default function MainLayout() {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 border-b border-slate-800/80 bg-slate-950/70 px-4 py-3 backdrop-blur md:hidden">
          <div className="mx-auto flex max-w-app items-center justify-between">
            <div>
              <p className="font-display text-base font-semibold text-white">SignFlow</p>
              <p className="text-xs text-slate-400">Assistant LSFB</p>
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto px-4 pb-24 pt-4 md:px-8 md:pb-8 md:pt-8">
          <div className="mx-auto w-full max-w-app">
            <Outlet />
          </div>
        </main>
      </div>
      <BottomNav />
    </div>
  );
}
