'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Card, Badge, Alert, Skeleton, Button } from '@/components/ui'
import { fetchAdminApplications, type AdminApplicationSummary } from '@/lib/api'
import { RefreshCw, ChevronRight, ChevronLeft } from 'lucide-react'

const PAGE_SIZE = 50

const STATUS_TABS = [
  { key: '', label: 'All' },
  { key: 'under_review', label: 'Under review' },
  { key: 'submitted', label: 'Submitted' },
  { key: 'decided', label: 'Decided' },
  { key: 'rejected', label: 'Rejected' },
]

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
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-stone-900">Applications queue</h1>
        <Button variant="ghost" size="sm" onClick={() => load()} icon={<RefreshCw className="w-4 h-4" />}>
          Refresh
        </Button>
      </div>

      <div className="flex flex-wrap gap-2">
        {STATUS_TABS.map(t => (
          <button
            key={t.key || 'all'}
            onClick={() => { setStatusFilter(t.key); setOffset(0) }}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
              statusFilter === t.key
                ? 'bg-stone-900 text-white'
                : 'bg-white text-stone-700 hover:bg-stone-100 border border-stone-200'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <Skeleton className="h-96 w-full" />
      ) : error ? (
        <Alert variant="error" message={error} />
      ) : items.length === 0 ? (
        <Card className="text-center py-12 text-stone-500">No applications match this filter.</Card>
      ) : (
        <Card className="bg-white shadow-soft overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-stone-50 text-xs uppercase tracking-wide text-stone-500">
              <tr>
                <Th>Application ID</Th>
                <Th>Phone</Th>
                <Th align="right">Loan ₹</Th>
                <Th>Purpose</Th>
                <Th>Status</Th>
                <Th>Decision</Th>
                <Th align="right">Score</Th>
                <Th>Submitted</Th>
                <Th />
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100">
              {items.map(a => (
                <tr key={a.application_id} className="hover:bg-stone-50">
                  <Td><span className="font-mono text-xs">{a.application_id}</span></Td>
                  <Td>{a.user_phone || '—'}</Td>
                  <Td align="right">{a.loan_amount.toLocaleString('en-IN')}</Td>
                  <Td>{a.loan_purpose}</Td>
                  <Td><StatusBadge status={a.status} /></Td>
                  <Td>
                    {a.decision ? (
                      <DecisionBadge decision={a.decision} />
                    ) : (
                      <span className="text-stone-400 text-xs">—</span>
                    )}
                  </Td>
                  <Td align="right">{a.score !== null ? (a.score * 100).toFixed(1) : '—'}</Td>
                  <Td><span className="text-xs text-stone-500">{new Date(a.submitted_at).toLocaleString()}</span></Td>
                  <Td>
                    <Link
                      href={`/admin/applications/${encodeURIComponent(a.application_id)}`}
                      className="text-harvest-600 hover:text-harvest-700 inline-flex items-center gap-1 text-xs"
                    >
                      Open <ChevronRight className="w-3 h-3" />
                    </Link>
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      <div className="flex items-center justify-between text-sm text-stone-600">
        <span>
          Showing {items.length === 0 ? 0 : offset + 1}-{offset + items.length} of {total}
        </span>
        <div className="flex gap-2">
          <Button
            variant="secondary" size="sm"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            icon={<ChevronLeft className="w-4 h-4" />}
          >
            Prev
          </Button>
          <Button
            variant="secondary" size="sm"
            disabled={offset + PAGE_SIZE >= total}
            onClick={() => setOffset(offset + PAGE_SIZE)}
            icon={<ChevronRight className="w-4 h-4" />}
            iconPosition="right"
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  )
}

function Th({ children, align = 'left' }: { children?: React.ReactNode; align?: 'left' | 'right' }) {
  return <th className={`px-4 py-3 text-${align} font-medium`}>{children}</th>
}
function Td({ children, align = 'left' }: { children?: React.ReactNode; align?: 'left' | 'right' }) {
  return <td className={`px-4 py-3 text-${align}`}>{children}</td>
}

function StatusBadge({ status }: { status: string }) {
  const variant = status === 'decided' || status === 'disbursed' ? 'success'
    : status === 'rejected' ? 'error'
    : status === 'under_review' ? 'warning'
    : 'neutral'
  return <Badge variant={variant}>{status.replace('_', ' ')}</Badge>
}

function DecisionBadge({ decision }: { decision: string }) {
  const variant = decision === 'approve' ? 'success'
    : decision === 'reject' ? 'error'
    : 'warning'
  return <Badge variant={variant}>{decision.replace('_', ' ')}</Badge>
}
