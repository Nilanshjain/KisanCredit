'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const TABS = [
  { href: '/admin',              label: 'Overview',     exact: true  },
  { href: '/admin/applications', label: 'Applications', exact: false },
  { href: '/admin/drift',        label: 'Drift',        exact: false },
]

export default function AdminTabs() {
  const pathname = usePathname()
  return (
    <nav className="flex items-center gap-1 -mb-px">
      {TABS.map(tab => {
        const active = tab.exact ? pathname === tab.href : pathname.startsWith(tab.href)
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className={`px-3 py-3 text-sm font-medium border-b-2 transition-colors ${
              active
                ? 'border-stone-900 text-stone-900'
                : 'border-transparent text-stone-500 hover:text-stone-900'
            }`}
          >
            {tab.label}
          </Link>
        )
      })}
    </nav>
  )
}
