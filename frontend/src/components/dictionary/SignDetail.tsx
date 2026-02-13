import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

import { apiBaseUrl } from "../../api/client";
import { deleteSign, listSignVideos, type Sign, type SignVideo, updateSign } from "../../api/signs";
import { MarkdownEditor } from "./MarkdownEditor";

interface SignDetailProps {
  sign: Sign;
  allSigns: Sign[];
  onSelectSign?: (signId: string) => void;
  onTrain?: (sign: Sign) => void;
  onUpdated?: (sign: Sign) => void;
  onDeleted?: (signId: string) => void;
}

export function SignDetail({
  sign,
  allSigns,
  onSelectSign,
  onTrain,
  onUpdated,
  onDeleted
}: SignDetailProps): JSX.Element {
  const [videos, setVideos] = useState<SignVideo[]>([]);
  const [activeVideoId, setActiveVideoId] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [draftDescription, setDraftDescription] = useState(sign.description ?? "");
  const [draftNotes, setDraftNotes] = useState(sign.notes ?? "");
  const [draftCategory, setDraftCategory] = useState(sign.category ?? "");
  const [draftTags, setDraftTags] = useState((sign.tags ?? []).join(", "));
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraftDescription(sign.description ?? "");
    setDraftNotes(sign.notes ?? "");
    setDraftCategory(sign.category ?? "");
    setDraftTags((sign.tags ?? []).join(", "));
    setIsEditing(false);
    setError(null);
  }, [sign]);

  useEffect(() => {
    let mounted = true;
    listSignVideos(sign.id)
      .then((items) => {
        if (!mounted) return;
        setVideos(items);
        if (items.length > 0) {
          setActiveVideoId((current) => current ?? items[0].id);
        } else {
          setActiveVideoId(null);
        }
      })
      .catch(() => {
        if (mounted) {
          setVideos([]);
          setActiveVideoId(null);
        }
      });

    return () => {
      mounted = false;
    };
  }, [sign.id]);

  const activeVideo = useMemo(() => videos.find((item) => item.id === activeVideoId) ?? null, [activeVideoId, videos]);
  const relatedById = useMemo(() => {
    const map = new Map<string, Sign>();
    for (const candidate of allSigns) {
      map.set(candidate.id, candidate);
    }
    return map;
  }, [allSigns]);

  const linkedSigns = useMemo(
    () =>
      sign.related_signs
        .map((id) => relatedById.get(id))
        .filter((item): item is Sign => Boolean(item)),
    [relatedById, sign.related_signs]
  );

  const save = async (): Promise<void> => {
    setIsSaving(true);
    setError(null);
    try {
      const payload = await updateSign(sign.id, {
        description: draftDescription,
        notes: draftNotes,
        category: draftCategory || undefined,
        tags: draftTags
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean)
      });
      onUpdated?.(payload);
      setIsEditing(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save sign.");
    } finally {
      setIsSaving(false);
    }
  };

  const remove = async (): Promise<void> => {
    const confirmed = window.confirm(`Delete sign "${sign.name}"?`);
    if (!confirmed) return;

    setError(null);
    try {
      await deleteSign(sign.id);
      onDeleted?.(sign.id);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Failed to delete sign.");
    }
  };

  return (
    <article className="card space-y-4 p-4">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="font-heading text-2xl">{sign.name}</h2>
          <p className="text-sm text-slate-400">{(sign.tags ?? []).map((tag) => `#${tag}`).join(" ")}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="touch-btn bg-accent text-slate-950" onClick={() => onTrain?.(sign)}>
            Train
          </button>
          {!isEditing ? (
            <button className="touch-btn bg-primary text-white" onClick={() => setIsEditing(true)}>
              Edit
            </button>
          ) : (
            <>
              <button className="touch-btn bg-secondary text-slate-950 disabled:opacity-60" disabled={isSaving} onClick={() => void save()}>
                {isSaving ? "Saving..." : "Save"}
              </button>
              <button className="touch-btn bg-slate-700 text-white" onClick={() => setIsEditing(false)}>
                Cancel
              </button>
            </>
          )}
          <button className="touch-btn bg-red-500 text-white" onClick={() => void remove()}>
            Delete
          </button>
        </div>
      </header>

      {error && <p className="rounded-btn bg-red-600/20 px-3 py-2 text-sm text-red-200">{error}</p>}

      <section>
        <h3 className="mb-2 font-heading text-lg">Reference Video</h3>
        {activeVideo ? (
          <video
            src={`${apiBaseUrl()}/api/v1/media/${activeVideo.id}/stream`}
            controls
            className="w-full rounded-card bg-black"
          />
        ) : (
          <p className="text-sm text-slate-400">No videos attached yet.</p>
        )}
      </section>

      <section>
        <h3 className="mb-2 font-heading text-lg">Description</h3>
        {isEditing ? (
          <textarea
            className="h-28 w-full rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-sm"
            value={draftDescription}
            onChange={(event) => setDraftDescription(event.target.value)}
          />
        ) : (
          <ReactMarkdown className="prose prose-invert max-w-none text-sm">{sign.description ?? "No description."}</ReactMarkdown>
        )}
      </section>

      <section>
        <h3 className="mb-2 font-heading text-lg">Notes</h3>
        {isEditing ? (
          <MarkdownEditor value={draftNotes} onChange={setDraftNotes} />
        ) : (
          <ReactMarkdown className="prose prose-invert max-w-none text-sm">{sign.notes ?? ""}</ReactMarkdown>
        )}
      </section>

      {isEditing && (
        <section className="grid gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-sm">
            Category
            <input
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-2"
              value={draftCategory}
              onChange={(event) => setDraftCategory(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Tags (comma separated)
            <input
              className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-2"
              value={draftTags}
              onChange={(event) => setDraftTags(event.target.value)}
            />
          </label>
        </section>
      )}

      <section>
        <h3 className="mb-2 font-heading text-lg">Linked Signs</h3>
        {linkedSigns.length === 0 ? (
          <p className="text-sm text-slate-400">No related signs.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {linkedSigns.map((related) => (
              <button
                type="button"
                key={related.id}
                className="rounded-full bg-slate-700/70 px-3 py-1 text-xs text-primary"
                onClick={() => onSelectSign?.(related.id)}
              >
                [[{related.name}]]
              </button>
            ))}
          </div>
        )}
      </section>

      <section>
        <h3 className="mb-2 font-heading text-lg">Videos ({videos.length})</h3>
        {videos.length === 0 ? (
          <p className="text-sm text-slate-400">No videos attached yet.</p>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {videos.map((video) => (
              <button
                key={video.id}
                type="button"
                className={`rounded-btn border p-2 text-left text-xs ${
                  video.id === activeVideoId ? "border-primary bg-primary/15" : "border-slate-700 bg-slate-900/50"
                }`}
                onClick={() => setActiveVideoId(video.id)}
              >
                <p className="font-mono text-[11px] uppercase tracking-wide text-slate-300">{video.type}</p>
                <p className="mt-1 text-slate-400">{(video.duration_ms / 1000).toFixed(1)}s</p>
                <p className="text-slate-500">{video.resolution}</p>
              </button>
            ))}
          </div>
        )}
      </section>

      <section>
        <h3 className="mb-2 font-heading text-lg">Stats</h3>
        <p className="text-sm text-slate-300">Training samples: {sign.training_sample_count}</p>
        <p className="text-sm text-slate-300">Model accuracy: {Math.round((sign.accuracy ?? 0) * 100)}%</p>
        <p className="text-sm text-slate-300">Usage count: {sign.usage_count}</p>
      </section>
    </article>
  );
}
