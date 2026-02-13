interface MarkdownEditorProps {
  value: string;
  onChange: (value: string) => void;
}

export function MarkdownEditor({ value, onChange }: MarkdownEditorProps): JSX.Element {
  return (
    <textarea
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="h-48 w-full rounded-btn border border-slate-700 bg-slate-900/60 p-3 text-sm"
      placeholder="Write notes with [[wikilinks]]"
      aria-label="Sign notes markdown editor"
    />
  );
}
