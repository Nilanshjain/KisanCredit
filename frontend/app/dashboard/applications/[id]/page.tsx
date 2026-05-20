'use client'

import { useEffect, useRef, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Card, Button, Badge, Alert, Skeleton } from '@/components/ui'
import { motion } from 'framer-motion'
import {
  ArrowLeft, RefreshCw, TrendingUp, FileText, CheckCircle2, XCircle,
  AlertTriangle, Clock, Loader2, User as UserIcon, Settings,
} from 'lucide-react'
import {
  getAccessToken, fetchTimeline, NON_TERMINAL_STATUSES,
  type ApplicationTimelineResponse, type ApplicationTimelineEvent,
} from '@/lib/api'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
const POLL_INTERVAL_MS = 3000

interface PredictionDetail {
  profitability_score: number
  decision: string
  confidence: number
  prediction_timestamp: string
  model_version?: string
  prediction_latency_ms?: number
}

interface ApplicationDetail {
  application_id: string
  user_id: string
  status: string
  loan_amount: number
  loan_purpose: string
  submitted_at: string
  processed_at?: string
  processing_time_ms?: number
  extracted_features?: Record<string, number>
  predictions: PredictionDetail[]
}

async function fetchDetail(applicationId: string): Promise<ApplicationDetail> {
  const token = getAccessToken()
  const response = await fetch(
    `${API_BASE_URL}/applications/${encodeURIComponent(applicationId)}`,
    {
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    },
  )
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || `detail fetch failed (${response.status})`)
  }
  return response.json()
}

type BadgeVariant = 'success' | 'error' | 'warning' | 'neutral'

const STATUS_META: Record<string, { label: string; variant: BadgeVariant; icon: typeof Clock }> = {
  submitted:    { label: 'Submitted',    variant: 'neutral', icon: Clock },
  under_review: { label: 'Under review', variant: 'warning', icon: Loader2 },
  decided:      { label: 'Decided',      variant: 'success', icon: CheckCircle2 },
  disbursed:    { label: 'Disbursed',    variant: 'success', icon: CheckCircle2 },
  rejected:     { label: 'Rejected',     variant: 'error',   icon: XCircle },
  approved:     { label: 'Approved',     variant: 'success', icon: CheckCircle2 },
}

function statusMeta(s: string): { label: string; variant: BadgeVariant; icon: typeof Clock } {
  return STATUS_META[s] ?? { label: s.toUpperCase(), variant: 'neutral', icon: Clock }
}

const DECISION_COPY: Record<string, { title: string; tone: string; icon: typeof CheckCircle2 }> = {
  approve:        { title: 'Approved',       tone: 'text-field-700',   icon: CheckCircle2 },
  manual_review:  { title: 'Manual review',  tone: 'text-harvest-700', icon: AlertTriangle },
  reject:         { title: 'Not approved',   tone: 'text-clay-700',    icon: XCircle },
}

export default function ApplicationDetailPage() {
  const params = useParams()
  const router = useRouter()
  const applicationId = params.id as string

  const [application, setApplication] = useState<ApplicationDetail | null>(null)
  const [timeline, setTimeline] = useState<ApplicationTimelineResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // Holds the interval id so we can clear it from outside the effect (refresh button)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const loadAll = async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const [tl, detail] = await Promise.all([
        fetchTimeline(applicationId),
        fetchDetail(applicationId).catch(() => null), // detail may briefly 404 right after submit; tolerate
      ])
      setTimeline(tl)
      if (detail) setApplication(detail)
      setError('')
    } catch (err) {
      if (!silent) setError(err instanceof Error ? err.message : 'Failed to load application')
    } finally {
      if (!silent) setLoading(false)
    }
  }

  useEffect(() => {
    loadAll()
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applicationId])

  // Start / stop polling based on whether the current status is terminal
  useEffect(() => {
    if (!timeline) return
    const isProcessing = NON_TERMINAL_STATUSES.has(timeline.current_status)
    if (isProcessing && !pollRef.current) {
      pollRef.current = setInterval(() => loadAll(true), POLL_INTERVAL_MS)
    }
    if (!isProcessing && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    return () => {
      // cleanup on status flip
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeline?.current_status])

  if (loading && !timeline) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 p-8">
        <div className="max-w-6xl mx-auto space-y-6">
          <Skeleton className="h-12 w-64" />
          <Skeleton className="h-48 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    )
  }

  if (error || !timeline) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 p-8">
        <div className="max-w-6xl mx-auto">
          <Alert variant="error" message={error || 'Application not found'} />
          <Button
            onClick={() => router.push('/dashboard')}
            className="mt-4"
            icon={<ArrowLeft className="w-4 h-4" />}
          >
            Back to dashboard
          </Button>
        </div>
      </div>
    )
  }

  const currentStatus = timeline.current_status
  const status = statusMeta(currentStatus)
  const StatusIcon = status.icon
  const isProcessing = NON_TERMINAL_STATUSES.has(currentStatus)
  const latestPrediction: PredictionDetail | null = application?.predictions?.[0] ?? null
  const decision = latestPrediction?.decision ?? null
  const decisionCopy = decision ? DECISION_COPY[decision] : null

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <Button
              onClick={() => router.push('/dashboard')}
              variant="ghost"
              icon={<ArrowLeft className="w-4 h-4" />}
            >
              Back
            </Button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Application</h1>
              <p className="text-gray-600 mt-1 font-mono text-sm">{applicationId}</p>
            </div>
          </div>

          <Button
            onClick={() => loadAll(false)}
            variant="outline"
            icon={<RefreshCw className="w-4 h-4" />}
          >
            Refresh
          </Button>
        </div>

        {/* Status card */}
        <Card className="bg-white shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-center">
            <div>
              <p className="text-sm text-gray-600 mb-2">Status</p>
              <div className="inline-flex items-center gap-2">
                <motion.span
                  animate={isProcessing ? { rotate: 360 } : {}}
                  transition={isProcessing ? { duration: 2, repeat: Infinity, ease: 'linear' } : {}}
                >
                  <StatusIcon className="w-5 h-5" />
                </motion.span>
                <Badge variant={status.variant}>{status.label}</Badge>
              </div>
            </div>

            <div>
              <p className="text-sm text-gray-600 mb-2">Loan amount</p>
              <p className="text-2xl font-bold text-gray-900">
                {application ? `₹${application.loan_amount.toLocaleString('en-IN')}` : '—'}
              </p>
            </div>

            <div>
              <p className="text-sm text-gray-600 mb-2">Score</p>
              {latestPrediction ? (
                <p className="text-2xl font-bold text-gray-900">
                  {Math.round(latestPrediction.profitability_score * 100)}/100
                </p>
              ) : (
                <p className="text-sm text-gray-400 italic">Pending decision…</p>
              )}
            </div>

            <div>
              <p className="text-sm text-gray-600 mb-2">Decision</p>
              {decisionCopy ? (
                <p className={`text-2xl font-bold ${decisionCopy.tone}`}>{decisionCopy.title}</p>
              ) : isProcessing ? (
                <p className="text-sm text-gray-400 italic">Processing…</p>
              ) : currentStatus === 'rejected' ? (
                <p className="text-2xl font-bold text-clay-700">Rejected</p>
              ) : (
                <p className="text-sm text-gray-400 italic">—</p>
              )}
            </div>
          </div>

          {isProcessing && (
            <Alert variant="info" className="mt-4">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Auto-refreshing every {POLL_INTERVAL_MS / 1000}s while processing…</span>
              </div>
            </Alert>
          )}
        </Card>

        {/* Lifecycle timeline */}
        <Card className="bg-white shadow-lg">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-amber-600" />
            Lifecycle
          </h2>

          {timeline.events.length === 0 ? (
            <p className="text-sm text-gray-500">No status events recorded yet.</p>
          ) : (
            <ol className="relative border-l-2 border-amber-200 pl-6 space-y-6">
              {timeline.events.map((evt, i) => (
                <TimelineRow key={`${evt.occurred_at}-${i}`} evt={evt} isLast={i === timeline.events.length - 1} />
              ))}
            </ol>
          )}
        </Card>

        {/* Decision panel — only once a prediction exists */}
        {latestPrediction && (
          <Card className="bg-white shadow-lg">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-amber-600" />
              ML prediction
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Stat label="Profitability score" value={`${(latestPrediction.profitability_score * 100).toFixed(1)}%`} tone="amber" />
              <Stat label="Model confidence"   value={`${(latestPrediction.confidence * 100).toFixed(1)}%`}            tone="blue" />
              <Stat label="Model version"      value={latestPrediction.model_version ?? '—'}                            tone="green" />
            </div>
            {latestPrediction.prediction_latency_ms !== undefined && (
              <p className="text-sm text-gray-500 mt-4">
                Inference completed in {latestPrediction.prediction_latency_ms.toFixed(2)} ms.
              </p>
            )}
          </Card>
        )}

        {/* Feature breakdown */}
        {application?.extracted_features && (
          <Card className="bg-white shadow-lg">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5 text-amber-600" />
              Extracted features (top 12)
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(application.extracted_features).slice(0, 12).map(([feature, value]) => (
                <div key={feature} className="border-l-4 border-amber-500 pl-4 py-2">
                  <p className="text-xs text-gray-600 truncate uppercase" title={feature}>
                    {feature.replace(/_/g, ' ')}
                  </p>
                  <p className="text-lg font-semibold text-gray-900">
                    {typeof value === 'number' ? value.toFixed(3) : value}
                  </p>
                </div>
              ))}
            </div>
            <p className="text-sm text-gray-500 mt-4">
              {Object.keys(application.extracted_features).length} total features extracted
            </p>
          </Card>
        )}
      </div>
    </div>
  )
}

function TimelineRow({ evt, isLast }: { evt: ApplicationTimelineEvent; isLast: boolean }) {
  const meta = statusMeta(evt.to_status)
  const Icon = meta.icon
  const ActorIcon = evt.actor_type === 'admin' ? Settings : evt.actor_type === 'user' ? UserIcon : RefreshCw
  return (
    <li className="relative">
      <span className="absolute -left-[34px] top-1 flex items-center justify-center w-7 h-7 rounded-full bg-white border-2 border-amber-300">
        <Icon className={`w-4 h-4 ${isLast ? 'text-amber-700' : 'text-amber-500'}`} />
      </span>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-semibold text-gray-900">
          {evt.from_status ? `${evt.from_status} → ` : ''}{evt.to_status}
        </span>
        <span className="text-xs text-gray-500">{new Date(evt.occurred_at).toLocaleString()}</span>
        <span className="inline-flex items-center gap-1 text-xs text-gray-500">
          <ActorIcon className="w-3 h-3" /> {evt.actor_type}
          {evt.actor_id ? ` (${evt.actor_id})` : ''}
        </span>
      </div>
      {evt.reason && (
        <p className="text-sm text-gray-600 mt-1 font-mono">{evt.reason}</p>
      )}
    </li>
  )
}

function Stat({ label, value, tone }: { label: string; value: string; tone: 'amber' | 'blue' | 'green' }) {
  const cls = {
    amber: 'from-amber-50 to-orange-50 text-amber-700',
    blue:  'from-blue-50 to-indigo-50 text-blue-700',
    green: 'from-green-50 to-emerald-50 text-green-700',
  }[tone]
  return (
    <div className={`p-4 bg-gradient-to-br ${cls.split(' ').slice(0, 2).join(' ')} rounded-lg`}>
      <p className="text-sm text-gray-600 mb-1">{label}</p>
      <p className={`text-3xl font-bold ${cls.split(' ').slice(2).join(' ')}`}>{value}</p>
    </div>
  )
}
