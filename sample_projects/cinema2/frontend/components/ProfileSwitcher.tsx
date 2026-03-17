"use client";

import { useState, useRef, useEffect } from "react";
import { useProfile } from "@/hooks/useProfile";
import { createProfile, deleteProfile as apiDeleteProfile } from "@/lib/api";

export default function ProfileSwitcher() {
  const { profile, setProfile, profiles, refreshProfiles } = useProfile();
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setCreating(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  async function handleCreate() {
    if (!newName.trim()) return;
    await createProfile(newName.trim());
    setProfile(newName.trim());
    setNewName("");
    setCreating(false);
    await refreshProfiles();
  }

  async function handleDelete(name: string) {
    await apiDeleteProfile(name);
    if (profile === name) setProfile(null);
    await refreshProfiles();
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 bg-film-card border border-film-border rounded-full px-3 py-1 text-sm hover:border-film-gold transition"
      >
        <span className="w-6 h-6 rounded-full bg-film-gold/20 flex items-center justify-center text-film-gold text-xs font-bold">
          {profile ? profile[0].toUpperCase() : "?"}
        </span>
        <span className="text-film-cream">{profile || "Select Profile"}</span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-56 bg-film-card border border-film-border rounded-lg shadow-xl overflow-hidden">
          {profiles.map((p) => (
            <div
              key={p}
              className="flex items-center justify-between px-3 py-2 hover:bg-film-dark cursor-pointer"
            >
              <button
                onClick={() => {
                  setProfile(p);
                  setOpen(false);
                }}
                className="flex-1 text-left text-sm text-film-cream"
              >
                {p}
                {p === profile && (
                  <span className="ml-2 text-film-gold text-xs">active</span>
                )}
              </button>
              <button
                onClick={() => handleDelete(p)}
                className="text-film-muted hover:text-red-400 text-xs ml-2"
              >
                ×
              </button>
            </div>
          ))}

          {creating ? (
            <div className="p-2 border-t border-film-border">
              <input
                autoFocus
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                placeholder="Profile name"
                className="w-full bg-film-dark border border-film-border rounded px-2 py-1 text-sm text-film-cream placeholder:text-film-muted focus:outline-none focus:border-film-gold"
              />
              <div className="flex gap-2 mt-1">
                <button
                  onClick={handleCreate}
                  className="text-xs text-film-gold hover:underline"
                >
                  Create
                </button>
                <button
                  onClick={() => setCreating(false)}
                  className="text-xs text-film-muted hover:underline"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setCreating(true)}
              className="w-full px-3 py-2 text-left text-sm text-film-gold hover:bg-film-dark border-t border-film-border"
            >
              + New Profile
            </button>
          )}
        </div>
      )}
    </div>
  );
}
