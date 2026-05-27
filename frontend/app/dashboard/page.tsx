'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { FileText, Plus, AlertCircle, ArrowRight, Loader2 } from 'lucide-react'
import { getUserApplications, type ApplicationSummary } from '@/lib/authApi'
import { useAuthStore } from '@/lib/authStore'
import { Button } from '@/components/ui'

// Status copy + dot color. We use a single dot rather than a full coloured
// pill — it's the same information with much less visual noise.
const STATUS_META: Record<string, { label: string; dot: string }> = {
  submitted:    { label: 'Submitted',    dot: 'bg-stone-400'   },
  under_review: { label: 'Under review', dot: 'bg-harvest-500' },
  processing:   { label: 'Processing',   dot: 'bg-harvest-500' },
  decided:      { label: 'Decided',      dot: 'bg-stone-900'   },
  approved:     { label: 'Approved',     dot: 'bg-field-600'   },
  rejected:     { label: 'Rejected',     dot: 'bg-clay-600'    },
  disbursed:    { label: 'Disbursed',    dot: 'bg-field-600'   },
}

export default function DashboardPage() {
  const router = useRouter()
  const { user, accessToken, isAuthenticated } = useAuthStore()

  const [applications, setApplications] = useState<ApplicationSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('')

  useEffect(() => {
    if (!isAuthenticated || !accessToken) router.push('/login')
  }, [isAuthenticated, accessToken, router])

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

  if (!isAuthenticated) return null

  return (
    <main className="min-h-screen bg-stone-50 pt-12 pb-16 px-6">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end justify-between gap-6">
          <div>
            <p className="text-sm text-stone-500">
              Signed in as {user?.full_name || user?.email}
            </p>
            <h1 className="mt-1 text-2xl font-semibold text-stone-900 tracking-tight">
              Your applications
            </h1>
          </div>
          <Link href="/apply">
            <Button variant="primary" size="md" icon={<Plus className="w-4 h-4" />}>
              New application
            </Button>
          </Link>
        </div>

        <div className="mt-8 flex items-center gap-3">
          <label className="text-sm text-stone-500">Filter</label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="text-sm bg-white border hairline rounded-md px-3 py-1.5 text-stone-900 focus:outline-none focus:ring-2 focus:ring-harvest-500/30 focus:border-harvest-500"
          >
            <option value="">All statuses</option>
            <option value="submitted">Submitted</option>
            <option value="under_review">Under review</option>
            <option value="decided">Decided</option>
            <option value="rejected">Rejected</option>
            <option value="disbursed">Disbursed</option>
          </select>
        </div>

        {error && (
          <div className="mt-6 flex items-start gap-3 rounded-md bg-clay-50 border hairline border-clay-200 p-4 text-sm">
            <AlertCircle className="w-4 h-4 text-clay-600 flex-shrink-0 mt-0.5" />
            <span className="text-clay-700">{error}</span>
          </div>
        )}

        {/* Applications list */}
        <div className="mt-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-16 text-stone-500 text-sm gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading applications…
            </div>
          ) : applications.length === 0 ? (
            <div className="text-center py-16 border hairline rounded-xl bg-white">
              <FileText className="w-6 h-6 text-stone-400 mx-auto mb-3" />
              <p className="text-stone-900 font-medium">No applications yet</p>
              <p className="mt-1 text-sm text-stone-500">
                Submit your first application to see it tracked here.
              </p>
              <div className="mt-6 flex justify-center">
                <Link href="/apply">
                  <Button variant="primary" size="md" icon={<Plus className="w-4 h-4" />}>
                    Start application
                  </Button>
                </Link>
              </div>
            </div>
          ) : (
            <ul className="divide-y hairline border hairline rounded-xl bg-white overflow-hidden">
              {applications.map((app) => {
                const meta = STATUS_META[app.status] ?? { label: app.status, dot: 'bg-stone-400' }
                return (
                  <li key={app.id}>
                    <Link
                      href={`/dashboard/applications/${app.application_id}`}
                      className="block px-5 py-4 hover:bg-stone-50 transition-colors"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2.5">
                            <span className={`w-1.5 h-1.5 rounded-full ${meta.dot}`} />
                            <span className="text-xs uppercase tracking-wider text-stone-500 font-medium">
                              {meta.label}
                            </span>
                            <span className="text-xs text-stone-400 font-numeric">
                              · {app.application_id}
                            </span>
                          </div>
                          <p className="mt-1 text-stone-900 font-medium truncate">
                            {app.loan_purpose}
                          </p>
                          <p className="mt-0.5 text-xs text-stone-500">
                            Submitted {new Date(app.submitted_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
                          </p>
                        </div>
                        <div className="text-right flex-shrink-0">
                          <p className="text-xs text-stone-500">Loan amount</p>
                          <p className="font-numeric font-semibold text-stone-900">
                            ₹{app.loan_amount.toLocaleString('en-IN')}
                          </p>
                        </div>
                        <ArrowRight className="w-4 h-4 text-stone-400 flex-shrink-0" />
                      </div>
                    </Link>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </div>
    </main>
  )
}
