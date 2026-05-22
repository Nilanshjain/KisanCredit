'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Sprout, FileText, Clock, CheckCircle, XCircle, AlertCircle, LogOut, User, Plus } from 'lucide-react'
import { getUserApplications, logout as apiLogout, type ApplicationSummary } from '@/lib/authApi'
import { useAuthStore } from '@/lib/authStore'

const STATUS_CONFIG = {
  submitted: {
    icon: Clock,
    color: 'bg-blue-100 text-blue-700 border-blue-300',
    label: 'Submitted'
  },
  processing: {
    icon: Clock,
    color: 'bg-amber-100 text-amber-700 border-amber-300',
    label: 'Processing'
  },
  approved: {
    icon: CheckCircle,
    color: 'bg-green-100 text-green-700 border-green-300',
    label: 'Approved'
  },
  rejected: {
    icon: XCircle,
    color: 'bg-red-100 text-red-700 border-red-300',
    label: 'Rejected'
  },
  disbursed: {
    icon: CheckCircle,
    color: 'bg-green-100 text-green-700 border-green-300',
    label: 'Disbursed'
  },
}

export default function DashboardPage() {
  const router = useRouter()
  const { user, accessToken, isAuthenticated, logout: storeLogout } = useAuthStore()

  const [applications, setApplications] = useState<ApplicationSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('')

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated || !accessToken) {
      router.push('/login')
    }
  }, [isAuthenticated, accessToken, router])

  // Fetch applications
  useEffect(() => {
    const fetchApplications = async () => {
      if (!accessToken) return

      try {
        setIsLoading(true)
        const response = await getUserApplications(statusFilter || undefined)
        setApplications(response.applications)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load applications')
      } finally {
        setIsLoading(false)
      }
    }

    fetchApplications()
  }, [accessToken, statusFilter])

  const handleLogout = async () => {
    if (accessToken) {
      try {
        await apiLogout(accessToken)
      } catch (err) {
        console.error('Logout error:', err)
      }
    }
    storeLogout()
    router.push('/')
  }

  if (!isAuthenticated) {
    return null // Will redirect
  }

  return (
    <div className="min-h-screen bg-rural-pattern relative">
      {/* Header */}
      <header className="bg-white shadow-soft border-b border-stone-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2">
              <Sprout className="w-8 h-8 text-harvest-500" />
              <span className="text-2xl font-bold text-stone-900">KisanCredit</span>
            </Link>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-stone-700">
                <User className="w-5 h-5" />
                <span className="font-semibold">{user?.full_name || user?.email}</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 text-clay-600 hover:bg-clay-50 rounded-lg transition-colors"
              >
                <LogOut className="w-5 h-5" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-stone-900 mb-2">
            Welcome back, {user?.full_name?.split(' ')[0] || 'User'}!
          </h1>
          <p className="text-stone-600">Track your loan applications and manage your profile</p>
        </div>

        {/* Actions */}
        <div className="mb-8 flex flex-wrap gap-4">
          <Link href="/apply" className="btn-primary flex items-center gap-2">
            <Plus className="w-5 h-5" />
            New Loan Application
          </Link>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="input-field min-w-[200px]"
          >
            <option value="">All Applications</option>
            <option value="submitted">Submitted</option>
            <option value="processing">Processing</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
            <option value="disbursed">Disbursed</option>
          </select>
        </div>

        {/* Error Message */}
        {error && (
          <div className="card bg-clay-50 border-clay-300 mb-6 p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-6 h-6 text-clay-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-bold text-clay-800 mb-1">Error</h3>
                <p className="text-clay-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Applications List */}
        {isLoading ? (
          <div className="text-center py-12">
            <div className="inline-block w-12 h-12 border-4 border-harvest-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="mt-4 text-stone-600">Loading your applications...</p>
          </div>
        ) : applications.length === 0 ? (
          <div className="card text-center py-12 p-8">
            <FileText className="w-16 h-16 text-stone-400 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-stone-900 mb-2">No Applications Yet</h3>
            <p className="text-stone-600 mb-6">
              You haven't submitted any loan applications. Start your first application now!
            </p>
            <Link href="/apply" className="btn-primary inline-flex items-center gap-2">
              <Plus className="w-5 h-5" />
              Apply for Loan
            </Link>
          </div>
        ) : (
          <div className="grid gap-4">
            {applications.map((app) => {
              const statusConfig = STATUS_CONFIG[app.status as keyof typeof STATUS_CONFIG] || STATUS_CONFIG.submitted
              const StatusIcon = statusConfig.icon

              return (
                <div key={app.id} className="card-hover p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <h3 className="text-lg font-bold text-stone-900">{app.loan_purpose}</h3>
                        <span className={`px-3 py-1 rounded-full text-sm font-semibold border-2 flex items-center gap-1 ${statusConfig.color}`}>
                          <StatusIcon className="w-4 h-4" />
                          {statusConfig.label}
                        </span>
                      </div>

                      <div className="grid md:grid-cols-2 gap-4 mb-4">
                        <div>
                          <p className="text-sm text-stone-500">Loan Amount</p>
                          <p className="text-xl font-bold text-stone-900">₹{app.loan_amount.toLocaleString('en-IN')}</p>
                        </div>
                        <div>
                          <p className="text-sm text-stone-500">Application ID</p>
                          <p className="text-sm font-mono text-stone-800">{app.application_id}</p>
                        </div>
                      </div>

                      <div className="flex gap-6 text-sm text-stone-600">
                        <div>
                          <span className="font-semibold">Submitted:</span> {new Date(app.submitted_at).toLocaleDateString('en-IN')}
                        </div>
                        {app.processed_at && (
                          <div>
                            <span className="font-semibold">Processed:</span> {new Date(app.processed_at).toLocaleDateString('en-IN')}
                          </div>
                        )}
                      </div>
                    </div>

                    <Link
                      href={`/dashboard/applications/${app.application_id}`}
                      className="btn-secondary"
                    >
                      View Details
                    </Link>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </main>
    </div>
  )
}
