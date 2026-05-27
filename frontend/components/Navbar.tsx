'use client'

import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import Button from './ui/Button'
import { Sprout, LogOut, LayoutDashboard, Building2 } from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'

export default function Navbar() {
  const router = useRouter()
  const { isAuthenticated, logout, user } = useAuthStore()
  const isAdmin = user?.role === 'admin'

  const handleLogout = () => {
    logout()
    router.push('/')
  }

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/85 backdrop-blur-md border-b hairline">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-2">
            <Sprout className="w-5 h-5 text-harvest-600" strokeWidth={2.25} />
            <span className="text-sm font-semibold text-stone-900 tracking-tight">
              KisanCredit
            </span>
          </Link>

          <div className="flex items-center gap-2">
            {isAuthenticated ? (
              <>
                <Link href={isAdmin ? '/admin' : '/dashboard'}>
                  <Button
                    variant="ghost"
                    size="sm"
                    icon={isAdmin
                      ? <Building2 className="w-4 h-4" />
                      : <LayoutDashboard className="w-4 h-4" />}
                  >
                    {isAdmin ? 'Operator console' : 'Dashboard'}
                  </Button>
                </Link>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleLogout}
                  icon={<LogOut className="w-4 h-4" />}
                >
                  Logout
                </Button>
              </>
            ) : (
              <Link href="/login">
                <Button variant="primary" size="sm">
                  Sign in
                </Button>
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
