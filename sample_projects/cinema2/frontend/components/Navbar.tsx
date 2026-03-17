"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import ProfileSwitcher from "./ProfileSwitcher";

export default function Navbar() {
  const [query, setQuery] = useState("");
  const router = useRouter();

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/?q=${encodeURIComponent(query.trim())}`);
    }
  }

  return (
    <nav className="sticky top-0 z-50 bg-film-black/95 backdrop-blur-md shadow-lg transition-all duration-300">
      <div className="max-w-7xl mx-auto px-4 h-18 flex items-center gap-8">
        <Link
          href="/"
          className="font-heading text-2xl font-extrabold text-film-gold tracking-tighter hover:text-white transition-colors"
        >
          PRIME<span className="text-white font-light">CINEMA</span>
        </Link>

        <div className="hidden md:flex gap-6 text-sm font-medium text-film-muted">
          <Link href="/" className="text-film-cream hover:text-film-gold transition-colors">
            Home
          </Link>
          <Link href="/profile" className="hover:text-film-cream transition-colors">
            My Taste
          </Link>
          <Link href="/blend" className="hover:text-film-cream transition-colors">
            Movie Night
          </Link>
        </div>

        <form onSubmit={handleSearch} className="ml-auto flex items-center gap-2 relative group">
          <div className="absolute left-3 text-film-muted/50 group-focus-within:text-film-gold transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A 7.5 7.5 0 1 0 5.196 5.196a 7.5 7.5 0 0 0 10.607 10.607z" />
            </svg>
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search..."
            className="bg-film-card border border-transparent focus:border-film-gold rounded-full pl-9 pr-4 py-2 text-sm text-film-cream placeholder:text-film-muted/70 focus:outline-none focus:ring-1 focus:ring-film-gold w-48 md:w-64 transition-all"
          />
        </form>

        <ProfileSwitcher />
      </div>
    </nav>
  );
}
