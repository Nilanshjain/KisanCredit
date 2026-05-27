'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Alert, Input } from '@/components/ui'
import {
  fetchAdminApplicationDetail, adminOverrideDecision,
  type AdminApplicationDetail,
} from '@/lib/api'
import { ArrowLeft, RefreshCw, Loader2, ShieldAlert } from 'lucide-react'

const STATUS_DOT: Record<string, string> = {
  submitted:    'bg-stone-400',
  under_review: 'bg-harvest-500',
  decided:      'bg-stone-900',
  approved:     'bg-field-600',
  rejected:     'bg-clay-600',
  disbursed:    'bg-field-600',
}

export default function AdminApplicationDetailPage() {
  const params = useParams()
  const router = useRouter()
  const applicationId = params.id as string

  const [data, setData] = useState<AdminApplicationDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [overrideDecision, setOverrideDecision] = useState<'approve' | 'reject' | 'manual_review'>('approve')
  const [overrideReason, setOverrideReason] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [overrideError, setOverrideError] = useState('')
  const [overrideSuccess, setOverrideSuccess] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      setData(await fetchAdminApplicationDetail(applicationId))
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [applicationId]) // eslint-disable-line react-hooks/exhaustive-deps

  const submitOverride = async () => {
    if (overrideReason.trim().length < 2) {
      setOverrideError('Reason must be at least 2 characters')
      return
    }
    setSubmitting(true)
    setOverrideError('')
    setOverrideSuccess('')
    try {
      await adminOverrideDecision(applicationId, overrideDecision, overrideReason.trim())
      setOverrideSuccess(`Decision overridden to ${overrideDecision}`)
      setOverrideReason('')
      await load()
    } catch (e) {
      setOverrideError(e instanceof Error ? e.message : 'Override failed')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-stone-500 py-16 justify-center">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading…
      </div>
    )
  }
  if (error || !data) {
    return (
      <div className="space-y-4">
        <Alert variant="error" message={error || 'Application not found'} />
        <button
          onClick={() => router.push('/admin/applications')}
          className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
        >
          <ArrowLeft className="w-3.5 h-3.5" /> Back to queue
        </button>
      </div>
    )
  }

  const pred = data.latest_prediction
  const canOverride = data.status === 'under_review' || data.status === 'decided' || data.status === 'submitted'
  const statusDot = STATUS_DOT[data.status] ?? 'bg-stone-400'

  return (
    <div className="space-y-10">
      <div className="flex items-center justify-between">
        <button
          onClick={() => router.push('/admin/applications')}
          className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
        >
          <ArrowLeft className="w-3.5 h-3.5" /> Queue
        </button>
        <button
          onClick={() => load()}
          className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      <header>
        <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">Application audit</p>
        <h1 className="mt-1 text-2xl font-semibold text-stone-900 tracking-tight font-numeric">
          {data.application_id}
        </h1>
      </header>

      {/* Applicant + status */}
      <section className="rounded-xl border hairline bg-white overflow-hidden">
        <div className="grid grid-cols-2 md:grid-cols-5 divide-x hairline">
          <KV label="Applicant" value={data.user_full_name || '—'} />
          <KV label="Phone" value={data.user_phone || '—'} numeric />
          <KV label="Loan amount" value={`₹${data.loan_amount.toLocaleString('en-IN')}`} numeric />
          <KV label="Purpose" value={data.loan_purpose} />
          <div className="px-5 py-4">
            <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">Status</p>
            <div className="mt-2 inline-flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${statusDot}`} />
              <span className="text-stone-900 font-medium">{data.status.replace('_', ' ')}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Latest prediction */}
      <section>
        <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
          Latest prediction
        </h2>
        <div className="rounded-xl border hairline bg-white p-6">
          {pred ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-4 font-numeric">
              <FieldStat label="Score" value={`${(pred.profitability_score * 100).toFixed(1)}`} unit="/ 100" />
              <FieldStat label="Confidence" value={`${(pred.confidence * 100).toFixed(1)}%`} />
              <FieldStat label="Decision" value={pred.decision.replace('_', ' ')} tone={pred.decision} />
              <FieldStat label="Model" value={pred.model_version || '—'} small />
              {typeof pred.prediction_latency_ms === 'number' && (
                <FieldStat label="Inference" value={`${pred.prediction_latency_ms.toFixed(0)} ms`} />
              )}
            </div>
          ) : (
            <p className="text-sm text-stone-500">No prediction yet — application still processing.</p>
          )}
        </div>
      </section>

      {/* Override */}
      <section>
        <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4 inline-flex items-center gap-2">
          <ShieldAlert className="w-3.5 h-3.5" /> Manual override
        </h2>
        <div className="rounded-xl border hairline bg-white p-6">
          {!canOverride ? (
            <p className="text-sm text-stone-500">
              This application is in status <code className="px-1 bg-stone-100 rounded font-numeric text-xs">{data.status}</code> and can't be overridden.
            </p>
          ) : (
            <div className="space-y-4">
              <div className="inline-flex bg-stone-100 rounded-md p-0.5">
                {(['approve', 'manual_review', 'reject'] as const).map(d => (
                  <button
                    key={d}
                    onClick={() => setOverrideDecision(d)}
                    className={`px-3 py-1.5 text-sm font-medium rounded transition-colors ${
                      overrideDecision === d
                        ? 'bg-white text-stone-900 shadow-soft'
                        : 'text-stone-500 hover:text-stone-900'
                    }`}
                  >
                    {d.replace('_', ' ')}
                  </button>
                ))}
              </div>
              <Input
                label="Reason (recorded on the audit event)"
                value={overrideReason}
                onChange={(e) => setOverrideReason(e.target.value)}
                placeholder="e.g. Phone-verified applicant; bank statements confirm stable income"
                helperText="Stored on application_status_events alongside your operator ID."
              />
              {overrideError && <Alert variant="error" message={overrideError} />}
              {overrideSuccess && <Alert variant="success" message={overrideSuccess} />}
              <button
                onClick={submitOverride}
                disabled={submitting}
                className="btn-base btn-primary inline-flex"
              >
                {submitting && <Loader2 className="w-4 h-4 animate-spin" />}
                Apply override
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Timeline */}
      <section>
        <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
          Lifecycle audit trail
        </h2>
        <div className="rounded-xl border hairline bg-white p-6">
          {data.timeline.length === 0 ? (
            <p className="text-sm text-stone-500">No events recorded.</p>
          ) : (
            <ol className="space-y-4 relative">
              <span className="absolute left-[5px] top-1.5 bottom-1.5 w-px bg-stone-200" />
              {data.timeline.map((e, i) => {
                const isLast = i === data.timeline.length - 1
                const dotColor = e.actor_type === 'admin'
                  ? 'bg-clay-600'
                  : isLast ? 'bg-stone-900' : 'bg-stone-400'
                return (
                  <li key={i} className="relative pl-5">
                    <span className={`absolute left-0 top-1 w-2.5 h-2.5 rounded-full border-2 border-white ${dotColor}`} />
                    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                      <span className="font-medium text-stone-900">
                        {e.from_status ? `${e.from_status} → ${e.to_status}` : e.to_status}
                      </span>
                      <span className="text-xs text-stone-500 font-numeric">
                        {new Date(e.occurred_at).toLocaleString()}
                      </span>
                      <span className={`text-xs font-medium ${e.actor_type === 'admin' ? 'text-clay-700' : 'text-stone-500'}`}>
                        · {e.actor_type}{e.actor_id ? ` (${e.actor_id})` : ''}
                      </span>
                    </div>
                    {e.reason && (
                      <p className="text-xs text-stone-500 mt-1 font-numeric">{e.reason}</p>
                    )}
                  </li>
                )
              })}
            </ol>
          )}
        </div>
      </section>

      {/* Features */}
      {data.extracted_features && (
        <section>
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
            Extracted features
          </h2>
          <div className="rounded-xl border hairline bg-white overflow-hidden">
            <div className="px-6 py-3 border-b hairline bg-stone-50 text-xs text-stone-500">
              {Object.keys(data.extracted_features).length} features
            </div>
            <div className="max-h-96 overflow-y-auto">
              <dl className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 font-numeric divide-y md:divide-y-0">
                {Object.entries(data.extracted_features).map(([k, v]) => (
                  <div key={k} className="px-5 py-3 odd:bg-stone-50/30 border-b hairline md:border-r last:border-r-0">
                    <dt className="text-[10px] text-stone-500 uppercase tracking-wider truncate" title={k}>
                      {k.replace(/_/g, ' ')}
                    </dt>
                    <dd className="mt-0.5 text-sm text-stone-900 font-medium">
                      {typeof v === 'number' ? v.toFixed(3) : v}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}

function KV({ label, value, numeric }: { label: string; value: string; numeric?: boolean }) {
  return (
    <div className="px-5 py-4">
      <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">{label}</p>
      <p className={`mt-2 text-stone-900 font-medium ${numeric ? 'font-numeric' : ''}`}>{value}</p>
    </div>
  )
}

function FieldStat({ label, value, unit, tone, small }: {
  label: string; value: string; unit?: string; tone?: string; small?: boolean
}) {
  const cls = tone === 'approve' ? 'text-field-700'
    : tone === 'reject' ? 'text-clay-700'
    : tone === 'manual_review' ? 'text-harvest-700'
    : 'text-stone-900'
  return (
    <div>
      <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">{label}</p>
      <p className={`mt-1.5 font-semibold ${cls} ${small ? 'text-sm' : 'text-lg'}`}>
        {value}
        {unit && <span className="text-xs text-stone-400 font-normal ml-1">{unit}</span>}
      </p>
    </div>
  )
}
