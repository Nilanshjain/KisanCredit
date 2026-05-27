'use client'

import { useEffect, useState } from 'react'
import { fetchAdminMetrics, type AdminMetricsOverview } from '@/lib/api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { RefreshCw, AlertTriangle, Loader2 } from 'lucide-react'

const REFRESH_MS = 10_000
const SMALL_SAMPLE_THRESHOLD = 30

export default function AdminOverview() {
  const [data, setData] = useState<AdminMetricsOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refreshing, setRefreshing] = useState(false)

  const load = async (silent = false) => {
    if (!silent) setLoading(true)
    setRefreshing(true)
    try {
      setData(await fetchAdminMetrics())
      setError('')
    } catch (e) {
      if (!silent) setError(e instanceof Error ? e.message : 'Failed to load metrics')
    } finally {
      if (!silent) setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    load()
    const id = setInterval(() => load(true), REFRESH_MS)
    return () => clearInterval(id)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-stone-500 py-16 justify-center">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading metrics…
      </div>
    )
  }
  if (error || !data) {
    return (
      <div className="rounded-md bg-clay-50 border hairline border-clay-200 p-4 text-sm text-clay-700">
        {error || 'No data'}
      </div>
    )
  }

  const decisionsByKey: Record<string, number> = Object.fromEntries(
    data.decisions.map(d => [d.decision, d.count]),
  )
  const overridesByKey: Record<string, number> = Object.fromEntries(
    data.override_decisions.map(d => [d.decision, d.count]),
  )

  const approvalRate = data.total_predictions > 0
    ? ((decisionsByKey.approve || 0) / data.total_predictions) * 100
    : null

  const overrideRate = (data.total_predictions + data.override_count) > 0
    ? (data.override_count / (data.total_predictions + data.override_count)) * 100
    : null

  const smallSample = data.total_predictions < SMALL_SAMPLE_THRESHOLD

  return (
    <div className="space-y-12">
      <header className="flex items-end justify-between gap-6">
        <div>
          <h1 className="text-2xl font-semibold text-stone-900 tracking-tight">Overview</h1>
          <p className="mt-1 text-sm text-stone-500">
            Last {data.window_hours}h · auto-refresh every {REFRESH_MS / 1000}s
          </p>
        </div>
        <button
          onClick={() => load(false)}
          className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </header>

      {/* Section 1: Model performance */}
      <section>
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500">
            Model performance (window)
          </h2>
          {smallSample && (
            <span className="inline-flex items-center gap-1.5 text-xs text-harvest-700">
              <AlertTriangle className="w-3.5 h-3.5" />
              Small sample (N = {data.total_predictions}) — percentiles are unreliable
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-stone-200 rounded-xl overflow-hidden border hairline">
          <Tile label="Model predictions" value={data.total_predictions.toLocaleString()} sublabel="overrides excluded" />
          <Tile
            label="Approval rate"
            value={approvalRate === null ? '—' : `${approvalRate.toFixed(1)}%`}
            sublabel={`${decisionsByKey.approve || 0} of ${data.total_predictions} approved by model`}
          />
          <Tile
            label="P95 latency"
            value={`${data.p95_latency_ms.toFixed(0)} ms`}
            sublabel={`avg ${data.avg_latency_ms.toFixed(0)} ms`}
            muted={smallSample}
          />
          <Tile
            label="Avg score"
            value={`${(data.avg_score * 100).toFixed(1)}`}
            unit="/ 100"
            sublabel="confidence weighted lower=riskier"
          />
        </div>

        <div className="mt-6 rounded-xl border hairline bg-white p-6">
          <div className="flex items-baseline justify-between mb-4">
            <p className="text-sm font-medium text-stone-900">Score distribution</p>
            <p className="text-xs text-stone-500">
              <span className="inline-block w-2 h-2 rounded-sm bg-clay-600 mr-1 align-middle" /> reject ‹ 0.4
              <span className="inline-block w-2 h-2 rounded-sm bg-harvest-500 ml-3 mr-1 align-middle" /> manual 0.4–0.6
              <span className="inline-block w-2 h-2 rounded-sm bg-field-600 ml-3 mr-1 align-middle" /> approve ≥ 0.6
            </p>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.score_histogram} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="#E7E5E4" vertical={false} />
                <XAxis
                  dataKey="range_low"
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}`}
                  tick={{ fill: '#78716C', fontSize: 11 }}
                  axisLine={{ stroke: '#E7E5E4' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#78716C', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
                />
                <Tooltip
                  cursor={{ fill: '#F5F5F4' }}
                  contentStyle={{ borderRadius: 8, border: '1px solid #E7E5E4', fontSize: 12 }}
                  formatter={((v: number) => [`${v}`, 'predictions']) as any}
                  labelFormatter={((low: number) => `${Math.round(low * 100)}–${Math.round((low + 0.1) * 100)}`) as any}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {data.score_histogram.map((bucket, i) => (
                    <Cell
                      key={i}
                      fill={
                        bucket.range_high <= 0.4 ? '#DC2626' :
                        bucket.range_low >= 0.6 ? '#16A34A' :
                        '#F59E0B'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="mt-6 grid grid-cols-3 gap-px bg-stone-200 rounded-xl overflow-hidden border hairline">
          <DecisionCell decision="approve"        count={decisionsByKey.approve       || 0} total={data.total_predictions} />
          <DecisionCell decision="manual_review"  count={decisionsByKey.manual_review || 0} total={data.total_predictions} />
          <DecisionCell decision="reject"         count={decisionsByKey.reject        || 0} total={data.total_predictions} />
        </div>
      </section>

      {/* Section 2: Operator activity */}
      <section>
        <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
          Operator activity (window)
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-stone-200 rounded-xl overflow-hidden border hairline">
          <Tile
            label="Overrides"
            value={data.override_count.toLocaleString()}
            sublabel="manual decisions by operators"
          />
          <Tile
            label="Override rate"
            value={overrideRate === null ? '—' : `${overrideRate.toFixed(1)}%`}
            sublabel="of all decisions in window"
          />
          <Tile
            label="→ approve"
            value={(overridesByKey.approve || 0).toLocaleString()}
            sublabel="model → approve overrides"
          />
          <Tile
            label="→ reject"
            value={(overridesByKey.reject || 0).toLocaleString()}
            sublabel="model → reject overrides"
          />
        </div>
      </section>

      {/* Section 3: Pipeline state (now snapshot, not window-bound) */}
      <section>
        <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
          Pipeline state <span className="ml-2 text-stone-400 normal-case font-normal tracking-normal">live snapshot, not window-bound</span>
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-stone-200 rounded-xl overflow-hidden border hairline">
          <Tile
            label="Pending review"
            value={data.pending_review_count.toLocaleString()}
            sublabel="status: submitted or under_review"
            tone={data.pending_review_count > 0 ? 'warning' : undefined}
          />
          <Tile
            label="Errored"
            value={data.errored_count.toLocaleString()}
            sublabel="inference failures, no prediction row"
            tone={data.errored_count > 0 ? 'warning' : undefined}
          />
          <Tile
            label="Drift baseline"
            value={data.drift_baseline_available ? 'present' : 'missing'}
            sublabel="training-set feature quantiles"
          />
          <Tile
            label="Drift buffer"
            value={data.recent_buffer_size.toLocaleString()}
            sublabel="recent inputs for PSI"
          />
        </div>
      </section>
    </div>
  )
}

function Tile({
  label, value, unit, sublabel, tone, muted,
}: {
  label: string
  value: string
  unit?: string
  sublabel?: string
  tone?: 'warning'
  muted?: boolean
}) {
  return (
    <div className="bg-white px-5 py-4">
      <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">{label}</p>
      <p className={`mt-2 text-2xl font-semibold font-numeric ${tone === 'warning' ? 'text-harvest-700' : muted ? 'text-stone-400' : 'text-stone-900'}`}>
        {value}
        {unit && <span className="text-sm text-stone-400 font-normal ml-1">{unit}</span>}
      </p>
      {sublabel && <p className="mt-1 text-[11px] text-stone-500">{sublabel}</p>}
    </div>
  )
}

function DecisionCell({
  decision, count, total,
}: { decision: string; count: number; total: number }) {
  const pct = total > 0 ? (count / total) * 100 : 0
  const label = decision === 'manual_review' ? 'Manual review' : decision.charAt(0).toUpperCase() + decision.slice(1)
  const tone = decision === 'approve' ? 'text-field-700' : decision === 'reject' ? 'text-clay-700' : 'text-harvest-700'
  return (
    <div className="bg-white px-5 py-4">
      <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">{label}</p>
      <p className={`mt-2 text-2xl font-semibold font-numeric ${tone}`}>{count}</p>
      <p className="mt-1 text-[11px] text-stone-500 font-numeric">{pct.toFixed(1)}% of model decisions</p>
    </div>
  )
}
