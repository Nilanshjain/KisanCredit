'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Card, Badge, Alert, Skeleton, Button, Input } from '@/components/ui'
import {
  fetchAdminApplicationDetail, adminOverrideDecision,
  type AdminApplicationDetail,
} from '@/lib/api'
import {
  ArrowLeft, RefreshCw, UserIcon, Phone, FileText, TrendingUp,
  ShieldAlert, CheckCircle2, XCircle, AlertTriangle, Clock,
} from 'lucide-react'

export default function AdminApplicationDetailPage() {
  const params = useParams()
  const router = useRouter()
  const applicationId = params.id as string

  const [data, setData] = useState<AdminApplicationDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // Override form state
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

  if (loading) return <Skeleton className="h-96 w-full" />
  if (error || !data) {
    return (
      <div className="space-y-4">
        <Alert variant="error" message={error || 'Application not found'} />
        <Button onClick={() => router.push('/admin/applications')} icon={<ArrowLeft className="w-4 h-4" />}>
          Back to queue
        </Button>
      </div>
    )
  }

  const pred = data.latest_prediction
  const canOverride = data.status === 'under_review' || data.status === 'decided' || data.status === 'submitted'

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <Button variant="ghost" onClick={() => router.push('/admin/applications')} icon={<ArrowLeft className="w-4 h-4" />}>
            Queue
          </Button>
          <div>
            <h1 className="text-2xl font-bold text-stone-900">Application audit</h1>
            <p className="font-mono text-xs text-stone-500 mt-1">{data.application_id}</p>
          </div>
        </div>
        <Button variant="ghost" size="sm" onClick={() => load()} icon={<RefreshCw className="w-4 h-4" />}>Refresh</Button>
      </div>

      {/* Applicant + status row */}
      <Card className="bg-white shadow-soft">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <KV label="Applicant" icon={<UserIcon className="w-3 h-3" />} value={data.user_full_name || '—'} />
          <KV label="Phone" icon={<Phone className="w-3 h-3" />} value={data.user_phone || '—'} mono />
          <KV label="Loan" value={`₹${data.loan_amount.toLocaleString('en-IN')}`} />
          <KV label="Purpose" value={data.loan_purpose} />
          <div>
            <p className="text-xs text-stone-500 mb-1">Status</p>
            <Badge variant={statusVariant(data.status)}>{data.status.replace('_', ' ')}</Badge>
          </div>
        </div>
      </Card>

      {/* Prediction */}
      <Card className="bg-white shadow-soft">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-harvest-600" /> Latest prediction
        </h2>
        {pred ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat label="Score" value={`${(pred.profitability_score * 100).toFixed(1)}%`} />
            <Stat label="Confidence" value={`${(pred.confidence * 100).toFixed(1)}%`} />
            <Stat label="Decision" value={pred.decision} tone={pred.decision} />
            <Stat label="Model" value={pred.model_version || '—'} />
          </div>
        ) : (
          <p className="text-sm text-stone-500">No prediction yet (application still processing).</p>
        )}
      </Card>

      {/* Override form */}
      <Card className="bg-white shadow-soft">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-clay-600" /> Admin override
        </h2>
        {!canOverride ? (
          <p className="text-sm text-stone-500">
            This application is in status <code className="px-1 bg-stone-100 rounded">{data.status}</code> and can't be overridden.
          </p>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {(['approve', 'manual_review', 'reject'] as const).map(d => (
                <button
                  key={d}
                  onClick={() => setOverrideDecision(d)}
                  className={`px-4 py-2 rounded-lg text-sm border transition-colors ${
                    overrideDecision === d
                      ? 'bg-stone-900 text-white border-stone-900'
                      : 'bg-white text-stone-700 border-stone-200 hover:bg-stone-50'
                  }`}
                >
                  {d.replace('_', ' ')}
                </button>
              ))}
            </div>
            <Input
              label="Reason (audit trail, required)"
              value={overrideReason}
              onChange={(e) => setOverrideReason(e.target.value)}
              placeholder="e.g. Manual verification confirmed steady employment"
              helperText="Will be stored on the application_status_events record alongside your admin ID."
            />
            {overrideError && <Alert variant="error" message={overrideError} />}
            {overrideSuccess && <Alert variant="success" message={overrideSuccess} />}
            <Button onClick={submitOverride} loading={submitting} variant="primary">
              Override decision
            </Button>
          </div>
        )}
      </Card>

      {/* Timeline */}
      <Card className="bg-white shadow-soft">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Clock className="w-5 h-5 text-harvest-600" /> Lifecycle audit trail
        </h2>
        {data.timeline.length === 0 ? (
          <p className="text-sm text-stone-500">No events recorded.</p>
        ) : (
          <ol className="relative border-l-2 border-stone-200 pl-6 space-y-4">
            {data.timeline.map((e, i) => {
              const Icon = e.to_status === 'rejected' ? XCircle
                : e.to_status === 'decided' || e.to_status === 'disbursed' ? CheckCircle2
                : e.to_status === 'under_review' ? AlertTriangle : Clock
              const tone = e.actor_type === 'admin' ? 'text-clay-700' : 'text-stone-700'
              return (
                <li key={i} className="relative">
                  <span className="absolute -left-[34px] top-1 flex items-center justify-center w-7 h-7 rounded-full bg-white border-2 border-stone-300">
                    <Icon className={`w-4 h-4 ${tone}`} />
                  </span>
                  <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <span className="font-medium text-stone-900">
                      {e.from_status ? `${e.from_status} → ${e.to_status}` : e.to_status}
                    </span>
                    <span className="text-xs text-stone-500">{new Date(e.occurred_at).toLocaleString()}</span>
                    <Badge variant={e.actor_type === 'admin' ? 'error' : 'neutral'}>
                      {e.actor_type}{e.actor_id ? ` · ${e.actor_id}` : ''}
                    </Badge>
                  </div>
                  {e.reason && <p className="text-sm text-stone-600 mt-1 font-mono">{e.reason}</p>}
                </li>
              )
            })}
          </ol>
        )}
      </Card>

      {/* Features */}
      {data.extracted_features && (
        <Card className="bg-white shadow-soft">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5 text-harvest-600" /> Extracted features
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-96 overflow-y-auto">
            {Object.entries(data.extracted_features).map(([k, v]) => (
              <div key={k} className="border-l-2 border-stone-200 pl-3 py-1">
                <p className="text-[10px] text-stone-500 uppercase tracking-wide truncate" title={k}>{k.replace(/_/g, ' ')}</p>
                <p className="text-sm font-mono text-stone-900">{typeof v === 'number' ? v.toFixed(3) : v}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-stone-500 mt-3">{Object.keys(data.extracted_features).length} features</p>
        </Card>
      )}
    </div>
  )
}

function KV({ label, value, icon, mono }: { label: string; value: string; icon?: React.ReactNode; mono?: boolean }) {
  return (
    <div>
      <p className="text-xs text-stone-500 mb-1 inline-flex items-center gap-1">{icon}{label}</p>
      <p className={`text-sm font-medium text-stone-900 ${mono ? 'font-mono' : ''}`}>{value}</p>
    </div>
  )
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  const cls = tone === 'approve' ? 'text-field-700'
    : tone === 'reject' ? 'text-clay-700'
    : tone === 'manual_review' ? 'text-harvest-700'
    : 'text-stone-900'
  return (
    <div className="p-3 bg-stone-50 rounded-lg">
      <p className="text-xs text-stone-500">{label}</p>
      <p className={`text-xl font-bold mt-1 ${cls}`}>{value}</p>
    </div>
  )
}

function statusVariant(s: string): 'success' | 'error' | 'warning' | 'neutral' {
  return s === 'decided' || s === 'disbursed' ? 'success'
    : s === 'rejected' ? 'error'
    : s === 'under_review' ? 'warning'
    : 'neutral'
}
