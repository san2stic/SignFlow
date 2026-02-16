import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import { Calendar, Loader2, Mail, Shield, User } from "lucide-react";

import { api } from "../lib/api";
import { useAuthStore } from "../stores/authStore";

export default function Profile(): JSX.Element {
  const user = useAuthStore((state) => state.user);
  const token = useAuthStore((state) => state.token);
  const updateUser = useAuthStore((state) => state.updateUser);
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [formData, setFormData] = useState({
    username: user?.username || "",
    full_name: user?.full_name || ""
  });

  const handleSubmit = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    try {
      if (!token) {
        throw new Error("Non authentifie");
      }
      const updatedUser = await api.updateProfile(token, formData);
      updateUser(updatedUser);
      setSuccess("Profil mis a jour.");
      setEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mise a jour impossible.");
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <div className="card p-5 text-sm text-text-secondary">
        Chargement du profil...
      </div>
    );
  }

  return (
    <section className="space-y-5">
      <header className="card p-5">
        <p className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Compte</p>
        <h1 className="mt-2 font-display text-2xl font-semibold text-white">Profil utilisateur</h1>
        <p className="mt-1 text-sm text-text-secondary">Gerez vos informations personnelles et etat de compte.</p>
      </header>

      <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
        <article className="card p-5">
          <div className="mb-6 flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-secondary text-slate-950">
              <User className="h-8 w-8" />
            </div>
            <div>
              <p className="font-display text-xl font-semibold text-white">{user.username}</p>
              <p className="text-sm text-text-secondary">{user.email}</p>
            </div>
          </div>

          {error && <div className="mb-4 rounded-btn border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}
          {success && <div className="mb-4 rounded-btn border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">{success}</div>}

          <form onSubmit={handleSubmit} className="space-y-4">
            <label className="block space-y-2 text-sm text-text-secondary">
              <span>Nom utilisateur</span>
              <input
                type="text"
                value={formData.username}
                onChange={(event) => setFormData({ ...formData, username: event.target.value })}
                disabled={!editing || loading}
                className="field-input disabled:cursor-not-allowed disabled:opacity-60"
              />
            </label>

            <label className="block space-y-2 text-sm text-text-secondary">
              <span>Nom complet</span>
              <input
                type="text"
                value={formData.full_name}
                onChange={(event) => setFormData({ ...formData, full_name: event.target.value })}
                disabled={!editing || loading}
                className="field-input disabled:cursor-not-allowed disabled:opacity-60"
              />
            </label>

            <div className="flex flex-wrap gap-2">
              {!editing ? (
                <button type="button" onClick={() => setEditing(true)} className="touch-btn bg-gradient-to-r from-primary to-secondary text-slate-950">
                  Modifier le profil
                </button>
              ) : (
                <>
                  <button type="submit" disabled={loading} className="touch-btn flex items-center gap-2 bg-gradient-to-r from-primary to-secondary text-slate-950 disabled:opacity-60">
                    {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                    Enregistrer
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setEditing(false);
                      setFormData({
                        username: user.username,
                        full_name: user.full_name || ""
                      });
                      setError("");
                      setSuccess("");
                    }}
                    disabled={loading}
                    className="touch-btn bg-slate-800 text-white"
                  >
                    Annuler
                  </button>
                </>
              )}
            </div>
          </form>
        </article>

        <aside className="space-y-3">
          <InfoCard icon={Mail} label="Email" value={user.email} />
          <InfoCard icon={Calendar} label="Membre depuis" value={new Date(user.created_at).toLocaleDateString("fr-FR")} />
          <InfoCard icon={Shield} label="Statut du compte" value={user.is_active ? "Actif" : "Inactif"} />
        </aside>
      </div>
    </section>
  );
}

function InfoCard({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }): JSX.Element {
  return (
    <div className="card p-4">
      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 text-primary-light">
        <Icon className="h-5 w-5" />
      </div>
      <p className="text-xs text-text-tertiary">{label}</p>
      <p className="mt-1 text-sm font-semibold text-white">{value}</p>
    </div>
  );
}
