import { ReactNode } from 'react'

interface MovieGridProps {
  children: ReactNode
  compact?: boolean
}

export default function MovieGrid({ children, compact }: MovieGridProps) {
  return (
    <div className={`grid gap-4 ${compact
      ? 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6'
      : 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5'
    }`}>
      {children}
    </div>
  )
}
