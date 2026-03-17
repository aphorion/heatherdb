"use client";

import { createContext, useContext } from "react";

export interface ProfileContextType {
  profile: string | null;
  setProfile: (name: string | null) => void;
  profiles: string[];
  setProfiles: (profiles: string[]) => void;
  refreshProfiles: () => Promise<void>;
}

export const ProfileContext = createContext<ProfileContextType>({
  profile: null,
  setProfile: () => {},
  profiles: [],
  setProfiles: () => {},
  refreshProfiles: async () => {},
});

export function useProfile() {
  return useContext(ProfileContext);
}
