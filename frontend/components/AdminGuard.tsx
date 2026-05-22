'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/lib/authStore'
import { getCurrentUser } from '@/lib/authApi'
import { Alert, Skeleton } from './ui'

interface Props {
  children: React.ReactNode
}

/** Client-side gate for /admin/* pages. Redirects non-admins to home.
 *  The backend require_admin() is the real gate; this is just UX so admins
 *  see content and non-admins see a friendly redirect instead of repeated 403s.
 */
export default function AdminGuard({ children }: Props) {
  const router = useRouter()
  const { accessToken, user, setUser, isAuthenticated } = useAuthStore()
  const [checking, setChecking] = useState(true)
  const [denied, setDenied] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function check() {
      if (!isAuthenticated || !accessToken) {
        router.push('/login?redirect=/admin')
        return
      }

      // Cached role check first (avoid hitting /users/me on every navigation)
      const cachedRole = user?.role
      if (cachedRole === 'admin') {
        setChecking(false)
        return
      }
      // If we have a user with a populated non-admin role, deny without a refetch.
      // (cachedRole is narrowed to 'user' | undefined here; truthy means 'user'.)
      if (user && cachedRole) {
        setDenied(true)
        setChecking(false)
        return
      }

      // No role on the cached user — fetch /users/me once to populate
      try {
        const fresh = await getCurrentUser()
        if (cancelled) return
        setUser(fresh)
        if (fresh.role !== 'admin') {
          setDenied(true)
        }
      } catch {
        if (!cancelled) setDenied(true)
      } finally {
        if (!cancelled) setChecking(false)
      }
    }

    check()
    return () => { cancelled = true }
  }, [accessToken, isAuthenticated, user, setUser, router])

  if (checking) {
    return (
      <div className="min-h-screen bg-stone-50 p-8">
        <div className="max-w-6xl mx-auto space-y-4">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-48 w-full" />
        </div>
      </div>
    )
  }

  if (denied) {
    return (
      <div className="min-h-screen bg-stone-50 p-8 flex items-center justify-center">
        <div className="max-w-md w-full">
          <Alert
            variant="error"
            title="Admin only"
            message="This area is reserved for lender/operator users. If you're the project owner, run scripts/promote_admin.py against your phone number."
          />
        </div>
      </div>
    )
  }

  return <>{children}</>
}
