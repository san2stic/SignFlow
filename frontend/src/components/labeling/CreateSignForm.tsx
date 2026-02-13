import { useState } from "react";
import { createSign } from "../../api/signs";

interface CreateSignFormProps {
  onSignCreated: (signId: string, signName: string) => void;
  onCancel: () => void;
}

export function CreateSignForm({ onSignCreated, onCancel }: CreateSignFormProps): JSX.Element {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const newSign = await createSign({ name: name.trim(), description: description.trim() || undefined });
      onSignCreated(newSign.id, newSign.name);
    } catch (err) {
      setError("Failed to create sign. It may already exist.");
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="sign-name" className="block text-sm font-medium text-slate-300 mb-1">
          Sign Name *
        </label>
        <input
          id="sign-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g., lsfb_bonjour"
          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-primary"
          required
          disabled={isSubmitting}
        />
      </div>

      <div>
        <label htmlFor="sign-description" className="block text-sm font-medium text-slate-300 mb-1">
          Description (optional)
        </label>
        <textarea
          id="sign-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Brief description of this sign..."
          rows={3}
          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-primary resize-none"
          disabled={isSubmitting}
        />
      </div>

      {error && (
        <div className="p-3 bg-red-900/20 border border-red-500/30 rounded text-red-300 text-sm">
          {error}
        </div>
      )}

      <div className="flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-slate-100 transition-colors"
          disabled={isSubmitting}
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!name.trim() || isSubmitting}
          className="px-4 py-2 text-sm font-medium bg-primary text-white rounded hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isSubmitting ? "Creating..." : "Create Sign"}
        </button>
      </div>
    </form>
  );
}
