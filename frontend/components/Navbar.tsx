'use client'

import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import Button from './ui/Button'
import { Sprout, LogOut } from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'

export default function Navbar() {
  const router = useRouter()
  const { isAuthenticated, logout } = useAuthStore()

  const handleLogout = () => {
    logout()
    router.push('/')
  }

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-stone-200">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
            >
              <Sprout className="w-8 h-8 text-field-600" />
            </motion.div>
            <span className="text-xl font-bold text-stone-900">
              KisanCredit
            </span>
          </Link>

          {/* Auth Buttons */}
          <div className="flex items-center gap-3">
            {isAuthenticated ? (
              <>
                <Link href="/dashboard">
                  <Button variant="ghost" size="sm">
                    Dashboard
                  </Button>
                </Link>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleLogout}
                  icon={<LogOut className="w-4 h-4" />}
                >
                  Logout
                </Button>
              </>
            ) : (
              <>
                <Link href="/login">
                  <Button variant="ghost" size="sm">
                    Login
                  </Button>
                </Link>
                <Link href="/signup">
                  <Button variant="primary" size="sm">
                    Sign Up
                  </Button>
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
