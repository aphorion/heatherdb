"use client";

import { useState, useEffect } from "react";
import { api, type Note, type EchoResponse } from "@/lib/api";
import FidelityBar from "./FidelityBar";
import ResultCard from "./ResultCard";

export default function EchoPanel() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EchoResponse | null>(null);

  useEffect(() => {
    api.notes().then((res) => setNotes(res.notes));
  }, []);

  async function handleEcho(noteId: number) {
    setSelectedId(noteId);
    setLoading(true);
    try {
      const res = await api.echo(noteId);
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  const filtered = search
    ? notes.filter((n) => n.text.toLowerCase().includes(search.toLowerCase()))
    : notes;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100 mb-1">Echo</h2>
        <p className="text-sm text-zinc-500">
          Select a note and see how the EAM remembers it. Other memories
          <em> interfere</em> during recall — revealing which notes are
          entangled in memory space.
        </p>
      </div>

      {/* Note selector */}
      <div className="space-y-2">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search notes..."
          className="w-full px-4 py-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 transition-colors"
        />
        <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
          {filtered.slice(0, 20).map((note) => (
            <button
              key={note.id}
              onClick={() => handleEcho(note.id)}
              className={`w-full text-left p-2.5 rounded-lg text-sm transition-colors ${
                selectedId === note.id
                  ? "bg-zinc-800 text-zinc-200 border border-zinc-700"
                  : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-300 border border-transparent"
              }`}
            >
              <span className="text-[10px] font-mono text-zinc-600 mr-2">
                #{note.id}
              </span>
              {note.text.length > 120
                ? note.text.slice(0, 120) + "..."
                : note.text}
            </button>
          ))}
          {filtered.length === 0 && (
            <p className="text-sm text-zinc-600 italic p-2">No notes found</p>
          )}
        </div>
      </div>

      {loading && (
        <p className="text-sm text-zinc-500">Echoing through memory...</p>
      )}

      {result && !loading && (
        <div className="space-y-5 animate-in fade-in duration-300">
          {/* Selected note */}
          <div className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg space-y-3">
            <p className="text-sm text-zinc-200 leading-relaxed">
              {result.note.text}
            </p>
            <FidelityBar value={result.fidelity} label="Recall fidelity" size="md" />
            <p className="text-xs text-zinc-500">
              {result.fidelity >= 0.8
                ? "High fidelity — this note is recalled almost perfectly. Few other memories interfere."
                : result.fidelity >= 0.5
                ? "Moderate fidelity — other memories are blending into the recall of this note."
                : "Low fidelity — this note is heavily distorted by interfering memories. It lives in a crowded region of memory space."}
            </p>
          </div>

          {/* Interfering memories */}
          {result.interfering.length > 0 && (
            <div className="space-y-3">
              <div>
                <h3 className="text-sm font-medium text-amber-400">
                  Interfering memories
                </h3>
                <p className="text-xs text-zinc-600 mt-0.5">
                  When the EAM reconstructs the selected note, these other
                  memories bleed in. Higher similarity means stronger
                  interference — these notes occupy overlapping regions of
                  memory space and have formed implicit associations.
                </p>
              </div>
              <div className="space-y-2">
                {result.interfering.map((r) => (
                  <ResultCard key={r.id} result={r} highlight />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
