'use client'

import { useEffect, useRef, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Button, Alert, Skeleton } from '@/components/ui'
import {
  ArrowLeft, RefreshCw, Loader2, Lightbulb,
} from 'lucide-react'
import {
  getAccessToken, fetchTimeline, NON_TERMINAL_STATUSES,
  fetchExplanation, fetchCounterfactual,
  type ApplicationTimelineResponse, type ApplicationTimelineEvent,
  type ApplicationExplanation, type CounterfactualResult,
} from '@/lib/api'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api/v1'
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

const STATUS_DOT: Record<string, string> = {
  submitted:    'bg-stone-400',
  under_review: 'bg-harvest-500',
  decided:      'bg-stone-900',
  approved:     'bg-field-600',
  rejected:     'bg-clay-600',
  disbursed:    'bg-field-600',
}

const STATUS_LABEL: Record<string, string> = {
  submitted:    'Submitted',
  under_review: 'Under review',
  decided:      'Decided',
  approved:     'Approved',
  rejected:     'Rejected',
  disbursed:    'Disbursed',
}

const DECISION_COPY: Record<string, { title: string; tone: string }> = {
  approve:        { title: 'Approved',     tone: 'text-field-700'   },
  manual_review:  { title: 'Manual review', tone: 'text-harvest-700' },
  reject:         { title: 'Not approved', tone: 'text-clay-700'    },
}

export default function ApplicationDetailPage() {
  const params = useParams()
  const router = useRouter()
  const applicationId = params.id as string

  const [application, setApplication] = useState<ApplicationDetail | null>(null)
  const [timeline, setTimeline] = useState<ApplicationTimelineResponse | null>(null)
  const [explanation, setExplanation] = useState<ApplicationExplanation | null>(null)
  const [counterfactual, setCounterfactual] = useState<CounterfactualResult | null>(null)
  const [language, setLanguage] = useState<'en' | 'hi'>('en')
  const [llmLoading, setLlmLoading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const loadAll = async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const [tl, detail] = await Promise.all([
        fetchTimeline(applicationId),
        fetchDetail(applicationId).catch(() => null),
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
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeline?.current_status])

  // The final poll fires timeline+detail in parallel; under concurrent commits
  // the timeline can show `decided` while the detail still observes
  // predictions: []. Polling has already stopped by then, so the empty state
  // sticks. One delayed re-fetch closes the gap.
  const postDecisionRefetchedRef = useRef(false)
  useEffect(() => {
    if (!timeline) return
    if (NON_TERMINAL_STATUSES.has(timeline.current_status)) {
      postDecisionRefetchedRef.current = false
      return
    }
    if (postDecisionRefetchedRef.current) return
    if (application?.predictions?.length) return
    postDecisionRefetchedRef.current = true
    const t = setTimeout(() => {
      fetchDetail(applicationId).then(setApplication).catch(() => {})
    }, 500)
    return () => clearTimeout(t)
  }, [timeline?.current_status, application?.predictions?.length, applicationId])

  // Once a decision exists, fetch the LLM explanation + counter-factual.
  // Refetches when language changes.
  useEffect(() => {
    const hasDecision = application?.predictions?.length && timeline?.current_status === 'decided'
    if (!hasDecision) return
    let cancelled = false
    setLlmLoading(true)
    Promise.all([
      fetchExplanation(applicationId, language),
      fetchCounterfactual(applicationId, language).catch(() => null),
    ])
      .then(([ex, cf]) => {
        if (cancelled) return
        setExplanation(ex)
        setCounterfactual(cf)
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLlmLoading(false) })
    return () => { cancelled = true }
  }, [applicationId, language, timeline?.current_status, application?.predictions?.length])

  if (loading && !timeline) {
    return (
      <main className="min-h-screen bg-stone-50 pt-12 px-6">
        <div className="max-w-3xl mx-auto space-y-4">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      </main>
    )
  }

  if (error || !timeline) {
    return (
      <main className="min-h-screen bg-stone-50 pt-12 px-6">
        <div className="max-w-3xl mx-auto">
          <Alert variant="error" message={error || 'Application not found'} />
          <div className="mt-4">
            <Button onClick={() => router.push('/dashboard')} variant="secondary" icon={<ArrowLeft className="w-4 h-4" />}>
              Back to dashboard
            </Button>
          </div>
        </div>
      </main>
    )
  }

  const currentStatus = timeline.current_status
  const isProcessing = NON_TERMINAL_STATUSES.has(currentStatus)
  const latestPrediction: PredictionDetail | null = application?.predictions?.[0] ?? null
  const decision = latestPrediction?.decision ?? null
  const decisionCopy = decision ? DECISION_COPY[decision] : null
  const statusDot = STATUS_DOT[currentStatus] ?? 'bg-stone-400'
  const statusLabel = STATUS_LABEL[currentStatus] ?? currentStatus

  return (
    <main className="min-h-screen bg-stone-50 pt-12 pb-16 px-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between gap-4 mb-8">
          <button
            onClick={() => router.push('/dashboard')}
            className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Back
          </button>
          <button
            onClick={() => loadAll(false)}
            className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>

        <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">Application</p>
        <h1 className="mt-1 text-2xl font-semibold text-stone-900 tracking-tight font-numeric">
          {applicationId}
        </h1>

        {/* Summary tiles */}
        <div className="mt-8 rounded-xl border hairline bg-white overflow-hidden">
          <div className="grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x hairline">
            <div className="px-5 py-4">
              <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">Status</p>
              <div className="mt-2 flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full ${statusDot}`} />
                <span className="font-medium text-stone-900">{statusLabel}</span>
              </div>
            </div>
            <div className="px-5 py-4">
              <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">Loan amount</p>
              <p className="mt-2 font-numeric font-semibold text-stone-900 text-lg">
                ₹{application?.loan_amount.toLocaleString('en-IN')}
              </p>
            </div>
            <div className="px-5 py-4">
              <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">Decision</p>
              <div className="mt-2">
                {decisionCopy ? (
                  <span className={`font-medium ${decisionCopy.tone}`}>{decisionCopy.title}</span>
                ) : isProcessing ? (
                  <span className="text-sm text-stone-500 inline-flex items-center gap-1.5">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" /> Processing
                  </span>
                ) : (
                  <span className="text-sm text-stone-500">—</span>
                )}
              </div>
            </div>
          </div>

          {latestPrediction && (
            <div className="px-5 py-3 border-t hairline bg-stone-50/50 text-xs text-stone-500 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-numeric">
              <span>Score <span className="text-stone-900 font-medium">{(latestPrediction.profitability_score * 100).toFixed(1)}</span> / 100</span>
              <span>Confidence <span className="text-stone-900 font-medium">{(latestPrediction.confidence * 100).toFixed(1)}%</span></span>
              <span>Model <span className="text-stone-900 font-medium">{latestPrediction.model_version ?? '—'}</span></span>
              {typeof latestPrediction.prediction_latency_ms === 'number' && (
                <span>Inference <span className="text-stone-900 font-medium">{latestPrediction.prediction_latency_ms.toFixed(0)}ms</span></span>
              )}
            </div>
          )}

          {isProcessing && (
            <div className="px-5 py-3 border-t hairline bg-stone-50 text-xs text-stone-500 inline-flex items-center gap-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Auto-refreshing every {POLL_INTERVAL_MS / 1000}s while processing.
            </div>
          )}
        </div>

        {/* Why this decision (NL explanation) */}
        {latestPrediction && (
          <section className="mt-10">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500">Why this decision</h2>
              <div className="inline-flex bg-stone-100 rounded-md p-0.5">
                <button
                  onClick={() => setLanguage('en')}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition ${language === 'en' ? 'bg-white text-stone-900 shadow-soft' : 'text-stone-500'}`}
                >EN</button>
                <button
                  onClick={() => setLanguage('hi')}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition ${language === 'hi' ? 'bg-white text-stone-900 shadow-soft' : 'text-stone-500'}`}
                >हिन्दी</button>
              </div>
            </div>
            <div className="rounded-xl border hairline bg-white p-6">
              {llmLoading && !explanation ? (
                <Skeleton className="h-16 w-full" />
              ) : explanation?.natural_language ? (
                <div className="space-y-4">
                  <p className="text-stone-800 leading-relaxed">{explanation.natural_language.text}</p>
                  {explanation.natural_language.suggestion && (
                    <div className="text-sm text-stone-600 leading-relaxed pl-4 border-l-2 border-harvest-400">
                      {explanation.natural_language.suggestion}
                    </div>
                  )}
                  <p className="text-xs text-stone-400">
                    {explanation.natural_language.source === 'gemini'
                      ? 'Generated by Gemini'
                      : 'Template (Gemini unavailable)'}
                    {explanation.natural_language.cached ? ' · cached' : ''}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-stone-500">Explanation will appear once the decision is finalised.</p>
              )}
            </div>
          </section>
        )}

        {/* Counter-factual */}
        {latestPrediction && counterfactual && counterfactual.changes.length > 0 && (
          <section className="mt-10">
            <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4 flex items-center gap-2">
              <Lightbulb className="w-3.5 h-3.5" /> How to improve
            </h2>
            <div className="rounded-xl border hairline bg-white p-6">
              {counterfactual.natural_language?.text && (
                <p className="text-stone-800 leading-relaxed mb-5">{counterfactual.natural_language.text}</p>
              )}
              <ul className="divide-y hairline border-t hairline border-b hairline -mx-6">
                {counterfactual.changes.map(c => (
                  <li key={c.feature} className="flex items-baseline justify-between gap-3 px-6 py-3">
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-stone-900">{c.display_label}</p>
                      <p className="text-xs text-stone-500 mt-0.5 font-numeric">
                        {c.display_unit}{c.current.toLocaleString('en-IN')}  →  {c.display_unit}{c.suggested.toLocaleString('en-IN')}
                      </p>
                    </div>
                    <div className="text-right shrink-0 font-numeric">
                      <p className="text-sm font-semibold text-field-700">+{(c.delta_score * 100).toFixed(1)} pts</p>
                      <p className="text-xs text-stone-500">→ {(c.new_score * 100).toFixed(0)} / 100</p>
                    </div>
                  </li>
                ))}
              </ul>
              <p className="text-xs text-stone-500 mt-5 leading-relaxed">
                {counterfactual.reachable
                  ? `These changes would push the score to ${(counterfactual.final_score * 100).toFixed(0)} / 100 — into the approve band.`
                  : 'The model couldn\'t identify a small enough set of changes to flip the decision into approve.'}
              </p>
            </div>
          </section>
        )}

        {/* Lifecycle */}
        <section className="mt-10">
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">Lifecycle</h2>
          <div className="rounded-xl border hairline bg-white p-6">
            {timeline.events.length === 0 ? (
              <p className="text-sm text-stone-500">No status events recorded yet.</p>
            ) : (
              <ol className="space-y-5 relative">
                {/* vertical line behind dots */}
                <span className="absolute left-[5px] top-1.5 bottom-1.5 w-px bg-stone-200" />
                {timeline.events.map((evt, i) => (
                  <TimelineRow key={`${evt.occurred_at}-${i}`} evt={evt} isLast={i === timeline.events.length - 1} />
                ))}
              </ol>
            )}
          </div>
        </section>

        {/* Extracted features */}
        {application?.extracted_features && (
          <section className="mt-10">
            <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
              Extracted features
            </h2>
            <div className="rounded-xl border hairline bg-white overflow-hidden">
              <div className="px-6 py-3 border-b hairline bg-stone-50 text-xs text-stone-500">
                {Object.keys(application.extracted_features).length} total — top 12 shown
              </div>
              <dl className="grid grid-cols-2 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x hairline font-numeric">
                {Object.entries(application.extracted_features).slice(0, 12).map(([feature, value]) => (
                  <div key={feature} className="px-5 py-3 odd:bg-stone-50/30">
                    <dt className="text-[11px] uppercase tracking-wider text-stone-500 truncate" title={feature}>
                      {feature.replace(/_/g, ' ')}
                    </dt>
                    <dd className="mt-0.5 text-stone-900 font-medium">
                      {typeof value === 'number' ? value.toFixed(3) : value}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </section>
        )}
      </div>
    </main>
  )
}

function TimelineRow({ evt, isLast }: { evt: ApplicationTimelineEvent; isLast: boolean }) {
  return (
    <li className="relative pl-5">
      <span className={`absolute left-0 top-1 w-2.5 h-2.5 rounded-full border-2 border-white ${isLast ? 'bg-stone-900' : 'bg-stone-400'}`} />
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
        <span className="font-medium text-stone-900">
          {evt.from_status ? `${evt.from_status} → ` : ''}{evt.to_status}
        </span>
        <span className="text-xs text-stone-500 font-numeric">
          {new Date(evt.occurred_at).toLocaleString()}
        </span>
        <span className="text-xs text-stone-500">· {evt.actor_type}</span>
      </div>
      {evt.reason && (
        <p className="text-xs text-stone-500 mt-1 font-numeric">{evt.reason}</p>
      )}
    </li>
  )
}
