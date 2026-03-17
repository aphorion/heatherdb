import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { api } from './api'
import BrowsePage from './pages/BrowsePage'
import ProfilePage from './pages/ProfilePage'
import RecommendPage from './pages/RecommendPage'
import ExplorePage from './pages/ExplorePage'
import BlendPage from './pages/BlendPage'

const NAV = [
  { to: '/', label: 'Browse', num: '01' },
  { to: '/recommend', label: 'For You', num: '02' },
  { to: '/profile', label: 'My List', num: '03' },
  { to: '/blend', label: 'Blend', num: '04' },
  { to: '/explore', label: 'Lab', num: '05' },
]

export default function App() {
  const [currentUser, setCurrentUser] = useState('')
  const [users, setUsers] = useState<string[]>([])
  const [newUser, setNewUser] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const location = useLocation()

  useEffect(() => { loadUsers() }, [])
  async function loadUsers() {
    try { setUsers((await api.listUsers()).users) } catch { setUsers([]) }
  }
  function createUser() {
    const name = newUser.trim().toLowerCase()
    if (!name) return
    setCurrentUser(name)
    if (!users.includes(name)) setUsers(prev => [...prev, name])
    setNewUser('')
  }
  async function deleteUser(name: string) {
    await api.deleteUser(name)
    setUsers(prev => prev.filter(u => u !== name))
    if (currentUser === name) setCurrentUser('')
  }

  const activeNav = NAV.find(n => n.to === location.pathname || (n.to === '/' && location.pathname === '/'))

  return (
    <div className="min-h-screen bg-film-black text-cream film-grain">
      {/* === SIDEBAR === */}
      <aside className={`fixed top-0 left-0 bottom-0 z-50 flex flex-col transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${
        sidebarOpen ? 'w-[220px]' : 'w-[60px]'
      }`}>
        {/* Gold accent line on left edge */}
        <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-gold/40 via-gold/10 to-transparent" />

        <div className="flex-1 flex flex-col bg-film-dark/80 backdrop-blur-xl border-r border-warm">
          {/* Logo */}
          <div className="h-16 flex items-center px-4 gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-8 h-8 rounded-full border border-gold/30 flex items-center justify-center group hover:bg-gold/10 transition-all shrink-0"
            >
              <div className="w-1.5 h-1.5 rounded-full bg-gold group-hover:shadow-[0_0_8px_rgba(232,179,75,0.5)] transition-all" />
            </button>
            {sidebarOpen && (
              <div className="animate-fade-in">
                <span className="font-display font-bold text-sm tracking-[0.15em] text-cream uppercase">Cinema</span>
              </div>
            )}
          </div>

          {/* Navigation */}
          <nav className="flex-1 py-6 px-2">
            <div className="space-y-1">
              {NAV.map(n => (
                <NavLink
                  key={n.to}
                  to={n.to}
                  end={n.to === '/'}
                  className={({ isActive }) =>
                    `group flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 relative overflow-hidden ${
                      isActive
                        ? 'bg-gold/[0.08]'
                        : 'hover:bg-cream/[0.02]'
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 bg-gold rounded-r-full" />
                      )}
                      <span className={`font-mono text-[10px] shrink-0 transition-colors ${
                        isActive ? 'text-gold' : 'text-cream-muted'
                      }`}>
                        {n.num}
                      </span>
                      {sidebarOpen && (
                        <span className={`font-display text-[13px] font-medium tracking-wide transition-colors ${
                          isActive ? 'text-cream' : 'text-cream-dim group-hover:text-cream'
                        }`}>
                          {n.label}
                        </span>
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </nav>

          {/* Profiles */}
          <div className="border-t border-warm px-3 py-4">
            {sidebarOpen && <div className="label mb-3 px-1">Profiles</div>}
            <div className="space-y-1.5">
              {users.map((u, i) => (
                <div key={u} className="group/u flex items-center gap-2">
                  <button
                    onClick={() => setCurrentUser(u)}
                    className={`flex items-center gap-2.5 flex-1 px-2 py-1.5 rounded-lg transition-all ${
                      u === currentUser ? 'bg-gold/[0.08]' : 'hover:bg-cream/[0.02]'
                    }`}
                    title={u}
                  >
                    <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-[10px] font-display font-bold uppercase shrink-0 transition-all ${
                      u === currentUser
                        ? 'bg-gold/20 text-gold border border-gold/30'
                        : 'bg-cream/[0.04] text-cream-dim border border-warm'
                    }`}>
                      {u.substring(0, 2)}
                    </div>
                    {sidebarOpen && (
                      <span className={`text-[12px] font-serif truncate ${u === currentUser ? 'text-cream' : 'text-cream-dim'}`}>{u}</span>
                    )}
                  </button>
                  {sidebarOpen && (
                    <button
                      onClick={() => deleteUser(u)}
                      className="opacity-0 group-hover/u:opacity-100 w-5 h-5 rounded flex items-center justify-center text-cream-muted hover:text-crimson transition-all shrink-0"
                    >
                      <svg width="10" height="10" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>
                    </button>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-3">
              {sidebarOpen ? (
                <div className="flex gap-1.5">
                  <input
                    className="flex-1 bg-cream/[0.03] border border-warm rounded-lg px-2.5 py-1.5 text-[11px] font-serif focus:outline-none focus:border-gold/20 text-cream placeholder-cream-muted transition-colors min-w-0"
                    placeholder="New profile..."
                    value={newUser}
                    onChange={e => setNewUser(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && createUser()}
                  />
                  <button onClick={createUser} className="w-7 h-7 rounded-lg border border-gold/20 hover:bg-gold/10 flex items-center justify-center text-gold transition-all shrink-0">
                    <svg width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><path d="M12 5v14m-7-7h14"/></svg>
                  </button>
                </div>
              ) : (
                <button onClick={() => setSidebarOpen(true)} className="w-full h-7 rounded-lg border border-warm hover:border-gold/20 flex items-center justify-center text-cream-muted hover:text-gold transition-all">
                  <svg width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><path d="M12 5v14m-7-7h14"/></svg>
                </button>
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* === MAIN === */}
      <main className={`min-h-screen transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${sidebarOpen ? 'ml-[220px]' : 'ml-[60px]'}`}>
        {/* Top bar */}
        <header className="sticky top-0 z-40 h-14 flex items-center justify-between px-8 bg-film-black/70 backdrop-blur-xl border-b border-warm">
          <div className="flex items-center gap-4">
            <span className="font-display text-[11px] font-bold tracking-[0.25em] text-cream-muted uppercase">
              {activeNav?.num}
            </span>
            <div className="w-4 h-[1px] bg-gold/30" />
            <h2 className="font-display text-sm font-semibold text-cream tracking-wide">
              {activeNav?.label || 'Cinema'}
            </h2>
          </div>
          {currentUser && (
            <div className="flex items-center gap-2.5">
              <div className="w-6 h-6 rounded-md bg-gold/15 border border-gold/20 flex items-center justify-center text-[9px] font-display font-bold uppercase text-gold">
                {currentUser.substring(0, 2)}
              </div>
              <span className="font-serif text-[12px] text-cream-dim italic">{currentUser}</span>
            </div>
          )}
        </header>

        <div className="min-h-[calc(100vh-3.5rem)]">
          <Routes>
            <Route path="/" element={<BrowsePage currentUser={currentUser} />} />
            <Route path="/profile" element={<ProfilePage currentUser={currentUser} />} />
            <Route path="/recommend" element={<RecommendPage currentUser={currentUser} />} />
            <Route path="/explore" element={<ExplorePage />} />
            <Route path="/blend" element={<BlendPage />} />
          </Routes>
        </div>

        <footer className="py-10 px-8 border-t border-warm">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[9px] text-cream-muted tracking-widest">HEATHERDB / SDM</span>
            <div className="flex items-center gap-2">
              <div className="w-1 h-1 rounded-full bg-gold/40 animate-flicker" />
              <span className="font-serif text-[11px] text-cream-muted italic">Elastic Associative Memory</span>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
