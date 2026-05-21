'use client'

import { useEffect, useState } from 'react'
import { Card, Alert, Skeleton, Button, Badge } from '@/components/ui'
import { fetchAdminDrift, type AdminDriftResponse, type AdminDriftFeature } from '@/lib/api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { RefreshCw, Activity } from 'lucide-react'

export default function AdminDriftPage() {
  const [data, setData] = useState<AdminDriftResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    try { setData(await fetchAdminDrift()); setError('') }
    catch (e) { setError(e instanceof Error ? e.message : 'Failed to load drift') }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  if (loading) return <Skeleton className="h-96 w-full" />
  if (error || !data) return <Alert variant="error" message={error || 'No data'} />

  const measurable = data.features.filter(f => f.psi !== null) as Array<AdminDriftFeature & { psi: number }>
  const chartData = measurable.slice(0, 20).map(f => ({
    feature: f.feature.length > 24 ? f.feature.slice(0, 22) + '…' : f.feature,
    full: f.feature, psi: f.psi, severity: f.severity,
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-stone-900">Input drift (PSI)</h1>
        <Button variant="ghost" size="sm" onClick={load} icon={<RefreshCw className="w-4 h-4" />}>Refresh</Button>
      </div>

      <Card className="bg-white shadow-soft">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Stat label="Stable" count={data.summary.stable || 0} tone="success" />
          <Stat label="Moderate" count={data.summary.moderate || 0} tone="warning" />
          <Stat label="Significant" count={data.summary.significant || 0} tone="error" />
          <Stat label="No baseline" count={(data.summary.unknown || 0) + (data.summary.no_data || 0)} tone="neutral" />
        </div>
        <div className="mt-4 text-xs text-stone-500 flex items-center gap-4 flex-wrap">
          <span><b>{data.n_recent_inputs}</b> recent inputs in buffer</span>
          <span>
            Baseline:{' '}
            {data.baseline_available
              ? <span className="text-field-700">✓ available (training quantiles)</span>
              : <span className="text-stone-500 italic">v2 only — current v1 model didn't ship quantiles</span>}
          </span>
        </div>
      </Card>

      {!data.baseline_available && (
        <Alert
          variant="info"
          title="Drift unavailable for v1"
          message="The synthetic-data v1 model didn't persist training feature quantiles. Phase 5 (Home Credit retrain) saves them, after which this chart becomes meaningful."
        />
      )}

      {chartData.length > 0 && (
        <Card className="bg-white shadow-soft">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-harvest-600" /> Top {chartData.length} features by drift
          </h2>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
                <XAxis type="number" domain={[0, 'auto']} tick={{ fill: '#78716c', fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#78716c', fontSize: 10 }} width={160} />
                <Tooltip
                  formatter={((v: number) => [v.toFixed(3), 'PSI']) as any}
                  labelFormatter={((_: unknown, payload: any) => payload?.[0]?.payload?.full || '') as any}
                />
                <ReferenceLine x={0.10} stroke="#eab308" strokeDasharray="3 3" label={{ value: 'moderate', position: 'top', fill: '#a16207', fontSize: 10 }} />
                <ReferenceLine x={0.25} stroke="#dc2626" strokeDasharray="3 3" label={{ value: 'significant', position: 'top', fill: '#b91c1c', fontSize: 10 }} />
                <Bar dataKey="psi" radius={[0, 4, 4, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={
                      d.severity === 'significant' ? '#dc2626'
                      : d.severity === 'moderate' ? '#eab308'
                      : '#16a34a'
                    } />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-stone-500 mt-2">
            PSI &lt; 0.10 stable · 0.10-0.25 moderate · &gt; 0.25 significant drift (consider retraining).
          </p>
        </Card>
      )}

      <Card className="bg-white shadow-soft p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-stone-50 text-xs uppercase tracking-wide text-stone-500">
            <tr>
              <th className="px-4 py-3 text-left">Feature</th>
              <th className="px-4 py-3 text-right">PSI</th>
              <th className="px-4 py-3 text-left">Severity</th>
              <th className="px-4 py-3 text-right">Samples</th>
              <th className="px-4 py-3 text-left">Baseline</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-100">
            {data.features.map(f => (
              <tr key={f.feature} className="hover:bg-stone-50">
                <td className="px-4 py-2 font-mono text-xs">{f.feature}</td>
                <td className="px-4 py-2 text-right font-mono">{f.psi !== null ? f.psi.toFixed(3) : '—'}</td>
                <td className="px-4 py-2">
                  <Badge variant={
                    f.severity === 'stable' ? 'success'
                    : f.severity === 'moderate' ? 'warning'
                    : f.severity === 'significant' ? 'error'
                    : 'neutral'
                  }>{f.severity}</Badge>
                </td>
                <td className="px-4 py-2 text-right">{f.n_current}</td>
                <td className="px-4 py-2 text-xs text-stone-500">{f.baseline_available ? '✓' : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function Stat({ label, count, tone }: { label: string; count: number; tone: 'success' | 'warning' | 'error' | 'neutral' }) {
  const cls = {
    success: 'bg-field-50 text-field-700 border-field-200',
    warning: 'bg-harvest-50 text-harvest-700 border-harvest-200',
    error:   'bg-clay-50 text-clay-700 border-clay-200',
    neutral: 'bg-stone-100 text-stone-700 border-stone-200',
  }[tone]
  return (
    <div className={`p-4 rounded-lg border ${cls}`}>
      <p className="text-xs uppercase tracking-wide">{label}</p>
      <p className="text-3xl font-bold mt-1">{count}</p>
      <p className="text-xs mt-1">features</p>
    </div>
  )
}
