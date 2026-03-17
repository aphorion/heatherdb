"use client";

import { useState, useEffect, useCallback } from "react";
import { api, type Note } from "@/lib/api";

interface NotesListProps {
  refreshKey?: number;
}

export default function NotesList({ refreshKey }: NotesListProps) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.notes();
      setNotes(res.notes);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  async function handleDelete(id: number) {
    await api.deleteNote(id);
    setNotes((prev) => prev.filter((n) => n.id !== id));
  }

  const filtered = filter
    ? notes.filter((n) =>
        n.text.toLowerCase().includes(filter.toLowerCase())
      )
    : notes;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-100 mb-1">Notes</h2>
          <p className="text-sm text-zinc-500">{notes.length} notes stored</p>
        </div>
        <button
          onClick={load}
          className="px-3 py-1.5 text-xs text-zinc-500 border border-zinc-800 rounded hover:border-zinc-600 hover:text-zinc-300 transition-colors"
        >
          Refresh
        </button>
      </div>

      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter notes..."
        className="w-full px-4 py-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 transition-colors"
      />

      {loading ? (
        <p className="text-sm text-zinc-600">Loading...</p>
      ) : filtered.length === 0 ? (
        <p className="text-sm text-zinc-600 italic">
          {filter ? "No notes match your filter" : "No notes yet. Write some!"}
        </p>
      ) : (
        <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
          {filtered.map((note) => (
            <div
              key={note.id}
              className="group p-3.5 bg-zinc-900/50 border border-zinc-800 rounded-lg hover:border-zinc-700 transition-colors"
            >
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm text-zinc-300 leading-relaxed flex-1">
                  {note.text}
                </p>
                <button
                  onClick={() => handleDelete(note.id)}
                  className="p-1 text-zinc-700 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition-all shrink-0"
                  title="Delete note"
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>
              <div className="mt-1.5 flex items-center gap-2">
                <span className="text-[10px] font-mono text-zinc-600">#{note.id}</span>
                <span className="text-[10px] text-zinc-700">
                  {new Date(note.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
