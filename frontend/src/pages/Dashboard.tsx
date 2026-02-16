import { Link } from "react-router-dom";

import { DashboardPage } from "./DashboardPage";
import { useAuthStore } from "../stores/authStore";

export default function Dashboard(): JSX.Element {
  const user = useAuthStore((state) => state.user);

  return (
    <div className="space-y-5">
      <section className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Vue globale</p>
            <h1 className="font-display text-2xl font-semibold text-white">Bienvenue {user?.username ?? "sur SignFlow"}</h1>
            <p className="mt-1 text-sm text-text-secondary">
              Suivez les performances du modele, les sessions d entrainement et l activite dictionnaire.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Link to="/translate" className="touch-btn bg-gradient-to-r from-primary to-secondary text-slate-950">
              Traduire en direct
            </Link>
            <Link to="/training" className="touch-btn bg-slate-800 text-white">
              Ouvrir entrainement
            </Link>
          </div>
        </div>
      </section>
      <DashboardPage />
    </div>
  );
}
