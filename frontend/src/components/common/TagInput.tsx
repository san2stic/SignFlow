import { useState } from "react";

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
}

export function TagInput({ tags, onChange }: TagInputProps): JSX.Element {
  const [value, setValue] = useState("");

  const addTag = (): void => {
    const next = value.trim();
    if (!next || tags.includes(next)) {
      return;
    }
    onChange([...tags, next]);
    setValue("");
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          className="w-full rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm"
          placeholder="Add tag"
        />
        <button className="touch-btn bg-primary text-white" onClick={addTag}>
          Add
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {tags.map((tag) => (
          <button
            key={tag}
            className="rounded-full bg-slate-700/60 px-3 py-1 text-xs"
            onClick={() => onChange(tags.filter((item) => item !== tag))}
          >
            #{tag}
          </button>
        ))}
      </div>
    </div>
  );
}
