"use client";

import { useState } from "react";
import { api, type WriteResponse, type Note } from "@/lib/api";
import FidelityBar from "./FidelityBar";

interface NoteInputProps {
  onNoteWritten?: () => void;
}

export default function NoteInput({ onNoteWritten }: NoteInputProps) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStarter, setLoadingStarter] = useState(false);
  const [result, setResult] = useState<WriteResponse | null>(null);
  const [recentNotes, setRecentNotes] = useState<Note[]>([]);

  async function handleWrite() {
    if (!text.trim() || loading) return;
    setLoading(true);
    try {
      const res = await api.write(text.trim());
      setResult(res);
      setText("");
      onNoteWritten?.();
      // Refresh recent notes
      const notes = await api.notes();
      setRecentNotes(notes.notes.slice(-5).reverse());
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadStarter() {
    setLoadingStarter(true);
    try {
      const res = await fetch("/starter_notes.txt");
      const content = await res.text();
      const lines = content
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l && !l.startsWith("#"));
      await api.load(lines);
      onNoteWritten?.();
      const notes = await api.notes();
      setRecentNotes(notes.notes.slice(-5).reverse());
    } finally {
      setLoadingStarter(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Write a Note</h2>
        <p className="text-sm text-zinc-500">
          Add observations, ideas, or facts. The database will find patterns you didn&apos;t
          encode.
        </p>
      </div>

      <div className="space-y-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Write an observation, idea, or fact..."
          className="w-full h-32 px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 resize-none transition-colors"
          onKeyDown={(e) => {
            if (e.key === "Enter" && e.metaKey) handleWrite();
          }}
        />
        <div className="flex items-center gap-3">
          <button
            onClick={handleWrite}
            disabled={!text.trim() || loading}
            className="px-5 py-2 bg-zinc-100 text-zinc-900 text-sm font-medium rounded-lg hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Writing..." : "Write"}
          </button>
          <span className="text-xs text-zinc-600">or Cmd+Enter</span>
          <div className="flex-1" />
          <button
            onClick={handleLoadStarter}
            disabled={loadingStarter}
            className="px-4 py-2 text-sm text-zinc-400 border border-zinc-800 rounded-lg hover:border-zinc-600 hover:text-zinc-300 disabled:opacity-40 transition-colors"
          >
            {loadingStarter ? "Loading..." : "Load starter notes"}
          </button>
        </div>
      </div>

      {result && (
        <div className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg space-y-2 animate-in fade-in duration-300">
          <p className="text-sm text-zinc-300">
            Note #{result.id} written.
          </p>
          <FidelityBar
            value={result.fidelity_hint}
            label="familiarity to your notebook"
            size="sm"
          />
        </div>
      )}

      {recentNotes.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
            Recent
          </h3>
          {recentNotes.map((note) => (
            <div
              key={note.id}
              className="p-3 bg-zinc-900/30 border border-zinc-800/50 rounded-lg"
            >
              <p className="text-sm text-zinc-400">{note.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
