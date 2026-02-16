import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Hand, Loader2 } from "lucide-react";

import { api } from "../lib/api";
import { useAuthStore } from "../stores/authStore";

export default function Login(): JSX.Element {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);

  const handleSubmit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await api.login({ email, password });
      setAuth(response.user, response.access_token);
      navigate("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connexion impossible.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="grid min-h-screen lg:grid-cols-[1.15fr_0.85fr]">
        <section className="relative hidden overflow-hidden border-r border-slate-800/80 p-10 lg:flex lg:flex-col">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_24%_24%,rgba(20,184,166,0.24),transparent_40%),radial-gradient(circle_at_78%_72%,rgba(6,182,212,0.24),transparent_44%)]" />
          <div className="relative z-10 flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-secondary text-slate-950">
              <Hand className="h-6 w-6" />
            </div>
            <div>
              <p className="font-display text-2xl font-bold text-white">SignFlow</p>
              <p className="text-sm text-text-secondary">Traduction LSFB en temps reel</p>
            </div>
          </div>

          <div className="relative z-10 mt-24 max-w-xl space-y-6">
            <p className="status-chip">Plateforme IA collaborative</p>
            <h1 className="font-display text-5xl font-semibold leading-tight text-white">
              Connectez-vous pour traduire, apprendre et deployer plus vite.
            </h1>
            <p className="text-base text-text-secondary">
              Pipeline live, dictionnaire graph, labeling et entrainement few-shot dans une seule interface.
            </p>
          </div>

          <div className="relative z-10 mt-auto grid grid-cols-3 gap-4">
            <Metric label="Precision modele" value="94.5%" />
            <Metric label="Signes suivis" value="1 200+" />
            <Metric label="Mode live" value="24/7" />
          </div>
        </section>

        <section className="flex items-center justify-center p-5 sm:p-8">
          <div className="w-full max-w-md space-y-6">
            <div className="lg:hidden">
              <div className="flex items-center justify-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-secondary text-slate-950">
                  <Hand className="h-5 w-5" />
                </div>
                <p className="font-display text-2xl font-semibold text-white">SignFlow</p>
              </div>
            </div>

            <div>
              <p className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Authentification</p>
              <h2 className="mt-2 font-display text-3xl font-semibold text-white">Connexion</h2>
              <p className="mt-1 text-sm text-text-secondary">Accedez a votre espace de traduction et d entrainement.</p>
            </div>

            {error && <div className="rounded-btn border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}

            <form onSubmit={handleSubmit} className="card space-y-4 p-5">
              <label className="block space-y-2 text-sm text-text-secondary">
                <span>Email</span>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="field-input"
                  placeholder="vous@exemple.com"
                  required
                  disabled={loading}
                />
              </label>

              <label className="block space-y-2 text-sm text-text-secondary">
                <span>Mot de passe</span>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="field-input"
                  placeholder="••••••••"
                  required
                  disabled={loading}
                />
              </label>

              <button
                type="submit"
                disabled={loading}
                className="touch-btn flex w-full items-center justify-center gap-2 bg-gradient-to-r from-primary to-secondary text-slate-950 disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Connexion...
                  </>
                ) : (
                  "Se connecter"
                )}
              </button>
            </form>

            <p className="text-center text-sm text-text-secondary">
              Pas encore de compte ?{" "}
              <Link to="/register" className="font-semibold text-primary-light hover:underline">
                Creer un compte
              </Link>
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="rounded-btn border border-slate-700/80 bg-slate-900/70 p-3">
      <p className="text-xs text-text-tertiary">{label}</p>
      <p className="mt-1 font-display text-xl font-semibold text-white">{value}</p>
    </div>
  );
}
