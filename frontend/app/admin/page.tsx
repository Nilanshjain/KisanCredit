'use client'

import { useEffect, useState } from 'react'
import { Card, Alert, Skeleton, Button } from '@/components/ui'
import { fetchAdminMetrics, type AdminMetricsOverview } from '@/lib/api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { TrendingUp, Activity, Clock, AlertCircle, RefreshCw, CheckCircle2 } from 'lucide-react'

const REFRESH_MS = 10_000

export default function AdminOverview() {
  const [data, setData] = useState<AdminMetricsOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      setData(await fetchAdminMetrics())
      setError('')
    } catch (e) {
      if (!silent) setError(e instanceof Error ? e.message : 'Failed to load metrics')
    } finally {
      if (!silent) setLoading(false)
    }
  }

  useEffect(() => {
    load()
    const id = setInterval(() => load(true), REFRESH_MS)
    return () => clearInterval(id)
  }, [])

  if (loading) return <Skeleton className="h-96 w-full" />
  if (error || !data) return <Alert variant="error" message={error || 'No data'} />

  const decisionsByKey: Record<string, number> = Object.fromEntries(
    data.decisions.map(d => [d.decision, d.count]),
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-stone-900">Operator overview</h1>
        <Button
          variant="ghost" size="sm" onClick={() => load(false)}
          icon={<RefreshCw className="w-4 h-4" />}
        >
          Refresh
        </Button>
      </div>

      <p className="text-sm text-stone-500">
        Last {data.window_hours}h · auto-refreshes every {REFRESH_MS / 1000}s
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatTile icon={<TrendingUp />} label="Predictions" value={data.total_predictions.toLocaleString()} sublabel={`avg score ${(data.avg_score * 100).toFixed(1)}%`} />
        <StatTile icon={<CheckCircle2 />} label="Approval rate" value={
          data.total_predictions > 0
            ? `${(((decisionsByKey.approve || 0) / data.total_predictions) * 100).toFixed(1)}%`
            : '—'
        } sublabel={`${decisionsByKey.approve || 0} / ${data.total_predictions}`} />
        <StatTile icon={<Clock />} label="P95 latency" value={`${data.p95_latency_ms.toFixed(0)} ms`} sublabel={`avg ${data.avg_latency_ms.toFixed(0)} ms`} />
        <StatTile
          icon={<AlertCircle />}
          label="Pending review"
          value={data.pending_review_count.toString()}
          sublabel={data.drift_baseline_available ? 'drift baseline ✓' : 'drift baseline: v2 only'}
          tone={data.pending_review_count > 0 ? 'warning' : 'neutral'}
        />
      </div>

      <Card className="bg-white shadow-soft">
        <h2 className="text-lg font-semibold text-stone-900 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-harvest-600" /> Score distribution
        </h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.score_histogram} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
              <XAxis
                dataKey="range_low"
                tickFormatter={(v) => `${(v * 100).toFixed(0)}`}
                label={{ value: 'Score bucket (0-100)', position: 'insideBottom', offset: -10 }}
                tick={{ fill: '#78716c', fontSize: 12 }}
              />
              <YAxis tick={{ fill: '#78716c', fontSize: 12 }} />
              <Tooltip
                formatter={((v: number) => [`${v} predictions`, 'Count']) as any}
                labelFormatter={((low: number) => `Score ${Math.round(low * 100)}-${Math.round((low + 0.1) * 100)}`) as any}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {data.score_histogram.map((bucket, i) => (
                  <Cell
                    key={i}
                    fill={
                      bucket.range_high <= 0.4 ? '#dc2626' :    // clay (reject band)
                      bucket.range_low >= 0.6 ? '#16a34a' :     // field (approve band)
                      '#eab308'                                  // harvest (manual review band)
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-stone-500 mt-2">
          Red = reject band (&lt;0.4), Yellow = manual review (0.4-0.6), Green = approve (&gt;0.6).
        </p>
      </Card>

      <Card className="bg-white shadow-soft">
        <h2 className="text-lg font-semibold text-stone-900 mb-4">Decision breakdown</h2>
        <div className="grid grid-cols-3 gap-4">
          <DecisionTile decision="approve" count={decisionsByKey.approve || 0} total={data.total_predictions} />
          <DecisionTile decision="manual_review" count={decisionsByKey.manual_review || 0} total={data.total_predictions} />
          <DecisionTile decision="reject" count={decisionsByKey.reject || 0} total={data.total_predictions} />
        </div>
      </Card>
    </div>
  )
}

function StatTile({ icon, label, value, sublabel, tone }: {
  icon: React.ReactNode; label: string; value: string; sublabel?: string; tone?: 'warning' | 'neutral'
}) {
  return (
    <Card className="bg-white shadow-soft">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-stone-500">{label}</p>
          <p className="text-3xl font-bold text-stone-900 mt-1">{value}</p>
          {sublabel && <p className="text-xs text-stone-500 mt-1">{sublabel}</p>}
        </div>
        <div className={`p-2 rounded-lg ${tone === 'warning' ? 'bg-harvest-50 text-harvest-700' : 'bg-stone-100 text-stone-600'}`}>
          {icon}
        </div>
      </div>
    </Card>
  )
}

function DecisionTile({ decision, count, total }: { decision: string; count: number; total: number }) {
  const pct = total > 0 ? (count / total) * 100 : 0
  const cls = decision === 'approve'
    ? 'bg-field-50 text-field-700 border-field-200'
    : decision === 'reject'
    ? 'bg-clay-50 text-clay-700 border-clay-200'
    : 'bg-harvest-50 text-harvest-700 border-harvest-200'
  return (
    <div className={`p-4 rounded-lg border ${cls}`}>
      <p className="text-xs uppercase tracking-wide">{decision.replace('_', ' ')}</p>
      <p className="text-2xl font-bold mt-1">{count}</p>
      <p className="text-xs mt-1">{pct.toFixed(1)}%</p>
    </div>
  )
}
