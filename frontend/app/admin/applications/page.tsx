'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { fetchAdminApplications, type AdminApplicationSummary } from '@/lib/api'
import { RefreshCw, ChevronRight, ChevronLeft, Loader2 } from 'lucide-react'

const PAGE_SIZE = 50

const STATUS_TABS = [
  { key: '',             label: 'All'          },
  { key: 'under_review', label: 'Under review' },
  { key: 'submitted',    label: 'Submitted'    },
  { key: 'decided',      label: 'Decided'      },
  { key: 'rejected',     label: 'Rejected'     },
]

const STATUS_DOT: Record<string, string> = {
  submitted:    'bg-stone-400',
  under_review: 'bg-harvest-500',
  decided:      'bg-stone-900',
  approved:     'bg-field-600',
  rejected:     'bg-clay-600',
  disbursed:    'bg-field-600',
}

export default function AdminApplicationsListPage() {
  const [items, setItems] = useState<AdminApplicationSummary[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [statusFilter, setStatusFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const data = await fetchAdminApplications(statusFilter || undefined, PAGE_SIZE, offset)
      setItems(data.applications)
      setTotal(data.total)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [statusFilter, offset]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-8">
      <header className="flex items-end justify-between gap-6">
        <div>
          <h1 className="text-2xl font-semibold text-stone-900 tracking-tight">Applications</h1>
          <p className="mt-1 text-sm text-stone-500">
            {total.toLocaleString()} total
          </p>
        </div>
        <button
          onClick={() => load()}
          className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </header>

      <div className="flex flex-wrap gap-1.5">
        {STATUS_TABS.map(t => (
          <button
            key={t.key || 'all'}
            onClick={() => { setStatusFilter(t.key); setOffset(0) }}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              statusFilter === t.key
                ? 'bg-stone-900 text-white'
                : 'bg-white text-stone-600 hover:text-stone-900 border hairline'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-sm text-stone-500 py-16 justify-center">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading…
        </div>
      ) : error ? (
        <div className="rounded-md bg-clay-50 border hairline border-clay-200 p-4 text-sm text-clay-700">
          {error}
        </div>
      ) : items.length === 0 ? (
        <div className="text-center py-16 border hairline rounded-xl bg-white text-stone-500 text-sm">
          No applications match this filter.
        </div>
      ) : (
        <div className="rounded-xl border hairline bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-stone-50 text-xs uppercase tracking-wider text-stone-500">
              <tr>
                <Th>Application</Th>
                <Th>Status</Th>
                <Th>Decision</Th>
                <Th align="right">Score</Th>
                <Th align="right">Loan ₹</Th>
                <Th>Submitted</Th>
                <Th />
              </tr>
            </thead>
            <tbody className="divide-y hairline">
              {items.map(a => (
                <tr key={a.application_id} className="hover:bg-stone-50 transition-colors">
                  <Td>
                    <div className="font-numeric text-xs text-stone-900 font-medium">{a.application_id}</div>
                    <div className="text-xs text-stone-500 truncate max-w-[200px]">{a.loan_purpose}</div>
                  </Td>
                  <Td>
                    <span className="inline-flex items-center gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full ${STATUS_DOT[a.status] ?? 'bg-stone-400'}`} />
                      <span className="text-stone-700">{a.status.replace('_', ' ')}</span>
                    </span>
                  </Td>
                  <Td>
                    {a.decision ? (
                      <DecisionPill decision={a.decision} />
                    ) : (
                      <span className="text-stone-400 text-xs">—</span>
                    )}
                  </Td>
                  <Td align="right">
                    <span className="font-numeric text-stone-900">
                      {a.score !== null ? (a.score * 100).toFixed(1) : '—'}
                    </span>
                  </Td>
                  <Td align="right">
                    <span className="font-numeric text-stone-900">
                      {a.loan_amount.toLocaleString('en-IN')}
                    </span>
                  </Td>
                  <Td>
                    <span className="text-xs text-stone-500 font-numeric">
                      {new Date(a.submitted_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}
                    </span>
                  </Td>
                  <Td align="right">
                    <Link
                      href={`/admin/applications/${encodeURIComponent(a.application_id)}`}
                      className="text-stone-500 hover:text-stone-900 inline-flex items-center gap-0.5 text-xs"
                    >
                      Open <ChevronRight className="w-3 h-3" />
                    </Link>
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex items-center justify-between text-sm text-stone-500">
        <span className="font-numeric">
          {items.length === 0 ? '0' : `${offset + 1}–${offset + items.length}`} of {total.toLocaleString()}
        </span>
        <div className="flex gap-2">
          <button
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            className="px-2.5 py-1 text-sm rounded-md border hairline bg-white text-stone-700 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-stone-50 inline-flex items-center gap-1"
          >
            <ChevronLeft className="w-3.5 h-3.5" /> Prev
          </button>
          <button
            disabled={offset + PAGE_SIZE >= total}
            onClick={() => setOffset(offset + PAGE_SIZE)}
            className="px-2.5 py-1 text-sm rounded-md border hairline bg-white text-stone-700 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-stone-50 inline-flex items-center gap-1"
          >
            Next <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  )
}

function Th({ children, align = 'left' }: { children?: React.ReactNode; align?: 'left' | 'right' }) {
  return <th className={`px-4 py-2.5 text-${align} font-medium`}>{children}</th>
}
function Td({ children, align = 'left' }: { children?: React.ReactNode; align?: 'left' | 'right' }) {
  return <td className={`px-4 py-3 text-${align} align-middle`}>{children}</td>
}

function DecisionPill({ decision }: { decision: string }) {
  const tone = decision === 'approve' ? 'text-field-700'
    : decision === 'reject' ? 'text-clay-700'
    : 'text-harvest-700'
  return <span className={`text-sm font-medium ${tone}`}>{decision.replace('_', ' ')}</span>
}
