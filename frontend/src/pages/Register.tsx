import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Hand, Loader2 } from "lucide-react";

import { api } from "../lib/api";
import { useAuthStore } from "../stores/authStore";

export default function Register(): JSX.Element {
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);

  const handleSubmit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setError("");

    if (password !== confirmPassword) {
      setError("Les mots de passe ne correspondent pas.");
      return;
    }

    if (password.length < 8) {
      setError("Le mot de passe doit contenir au moins 8 caracteres.");
      return;
    }

    setLoading(true);
    try {
      const response = await api.register({
        email,
        username,
        password,
        full_name: fullName || undefined
      });
      setAuth(response.user, response.access_token);
      navigate("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inscription impossible.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="grid min-h-screen lg:grid-cols-[0.95fr_1.05fr]">
        <section className="flex items-center justify-center p-5 sm:p-8">
          <div className="w-full max-w-lg space-y-6">
            <div className="lg:hidden">
              <div className="flex items-center justify-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-secondary text-slate-950">
                  <Hand className="h-5 w-5" />
                </div>
                <p className="font-display text-2xl font-semibold text-white">SignFlow</p>
              </div>
            </div>

            <div>
              <p className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Nouvel espace</p>
              <h1 className="mt-2 font-display text-3xl font-semibold text-white">Creer un compte</h1>
              <p className="mt-1 text-sm text-text-secondary">Configurez votre profil pour commencer la traduction en direct.</p>
            </div>

            {error && <div className="rounded-btn border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}

            <form onSubmit={handleSubmit} className="card space-y-4 p-5">
              <Field label="Email">
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
              </Field>

              <Field label="Nom utilisateur">
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  className="field-input"
                  placeholder="bastien"
                  minLength={3}
                  required
                  disabled={loading}
                />
              </Field>

              <Field label="Nom complet (optionnel)">
                <input
                  id="fullName"
                  type="text"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  className="field-input"
                  placeholder="Bastien Javaux"
                  disabled={loading}
                />
              </Field>

              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Mot de passe">
                  <input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    className="field-input"
                    placeholder="••••••••"
                    minLength={8}
                    required
                    disabled={loading}
                  />
                </Field>
                <Field label="Confirmer">
                  <input
                    id="confirmPassword"
                    type="password"
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    className="field-input"
                    placeholder="••••••••"
                    required
                    disabled={loading}
                  />
                </Field>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="touch-btn flex w-full items-center justify-center gap-2 bg-gradient-to-r from-primary to-secondary text-slate-950 disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Creation...
                  </>
                ) : (
                  "Creer mon compte"
                )}
              </button>
            </form>

            <p className="text-center text-sm text-text-secondary">
              Deja inscrit ?{" "}
              <Link to="/login" className="font-semibold text-primary-light hover:underline">
                Se connecter
              </Link>
            </p>
          </div>
        </section>

        <section className="relative hidden overflow-hidden border-l border-slate-800/80 p-10 lg:flex lg:flex-col">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_76%_26%,rgba(20,184,166,0.24),transparent_42%),radial-gradient(circle_at_18%_72%,rgba(6,182,212,0.24),transparent_44%)]" />
          <div className="relative z-10 flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-secondary text-slate-950">
              <Hand className="h-6 w-6" />
            </div>
            <div>
              <p className="font-display text-2xl font-bold text-white">SignFlow</p>
              <p className="text-sm text-text-secondary">Concevoir, tester et deployer</p>
            </div>
          </div>

          <div className="relative z-10 mt-24 max-w-xl space-y-6">
            <p className="status-chip">Onboarding produit</p>
            <h2 className="font-display text-5xl font-semibold leading-tight text-white">
              Lancez votre environnement de traduction en quelques minutes.
            </h2>
            <p className="text-base text-text-secondary">
              Acces direct aux modules dictionnaire, labeling et entrainement avec suivi de performance modele.
            </p>
          </div>

          <ul className="relative z-10 mt-auto space-y-3 rounded-card border border-slate-700/80 bg-slate-900/65 p-5 text-sm text-text-secondary">
            <li>Configuration initiale guidee.</li>
            <li>Historique live des sessions et predictions.</li>
            <li>Workflow complet de labels et deployement.</li>
          </ul>
        </section>
      </div>
    </main>
  );
}

function Field({ label, children }: { label: string; children: JSX.Element }): JSX.Element {
  return (
    <label className="block space-y-2 text-sm text-text-secondary">
      <span>{label}</span>
      {children}
    </label>
  );
}
