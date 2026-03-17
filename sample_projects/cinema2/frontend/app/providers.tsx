"use client";

import { useState, useEffect, useCallback, type ReactNode } from "react";
import { ProfileContext } from "@/hooks/useProfile";
import { getProfiles } from "@/lib/api";

export function Providers({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<string[]>([]);

  const setProfile = useCallback((name: string | null) => {
    setProfileState(name);
    if (name) {
      localStorage.setItem("cinema2_profile", name);
    } else {
      localStorage.removeItem("cinema2_profile");
    }
  }, []);

  const refreshProfiles = useCallback(async () => {
    try {
      const list = await getProfiles();
      setProfiles(list);
    } catch {
      // backend not ready
    }
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem("cinema2_profile");
    if (saved) setProfileState(saved);
    refreshProfiles();
  }, [refreshProfiles]);

  return (
    <ProfileContext.Provider
      value={{ profile, setProfile, profiles, setProfiles, refreshProfiles }}
    >
      {children}
    </ProfileContext.Provider>
  );
}
