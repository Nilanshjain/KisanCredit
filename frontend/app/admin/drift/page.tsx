'use client'

import { useEffect, useState } from 'react'
import { Alert } from '@/components/ui'
import { fetchAdminDrift, type AdminDriftResponse, type AdminDriftFeature } from '@/lib/api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { RefreshCw, Loader2, AlertTriangle } from 'lucide-react'

const PSI_SMALL_SAMPLE = 30

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

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-stone-500 py-16 justify-center">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading drift…
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

  const measurable = data.features.filter(f => f.psi !== null) as Array<AdminDriftFeature & { psi: number }>
  const chartData = measurable
    .slice()
    .sort((a, b) => b.psi - a.psi)
    .slice(0, 20)
    .map(f => ({
      feature: f.feature.length > 24 ? f.feature.slice(0, 22) + '…' : f.feature,
      full: f.feature, psi: f.psi, severity: f.severity,
    }))

  const smallSample = data.n_recent_inputs < PSI_SMALL_SAMPLE

  return (
    <div className="space-y-10">
      <header className="flex items-end justify-between gap-6">
        <div>
          <h1 className="text-2xl font-semibold text-stone-900 tracking-tight">Input drift</h1>
          <p className="mt-1 text-sm text-stone-500 font-numeric">
            {data.n_recent_inputs} recent input{data.n_recent_inputs === 1 ? '' : 's'} in buffer · Population Stability Index vs training baseline
          </p>
        </div>
        <button
          onClick={load}
          className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </header>

      {smallSample && data.baseline_available && (
        <div className="rounded-md bg-harvest-50 border hairline border-harvest-200 p-4 text-sm text-harvest-800 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Small sample</p>
            <p className="mt-1 text-harvest-700">
              PSI needs roughly {PSI_SMALL_SAMPLE}+ recent inputs to be informative. With {data.n_recent_inputs}{' '}
              {data.n_recent_inputs === 1 ? 'sample' : 'samples'} the per-feature scores below will look like extreme drift even when nothing is wrong.
            </p>
          </div>
        </div>
      )}

      {!data.baseline_available && (
        <Alert
          variant="info"
          title="Drift baseline missing"
          message="The current model didn't ship training-set feature quantiles. Drift can't be measured until a model trained with the quantile-export step is loaded."
        />
      )}

      {/* Summary tiles */}
      <section>
        <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
          Summary
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-stone-200 rounded-xl overflow-hidden border hairline">
          <DriftTile label="Stable"      count={data.summary.stable || 0}      tone="field"   />
          <DriftTile label="Moderate"    count={data.summary.moderate || 0}    tone="harvest" />
          <DriftTile label="Significant" count={data.summary.significant || 0} tone="clay"    />
          <DriftTile label="No baseline" count={(data.summary.unknown || 0) + (data.summary.no_data || 0)} tone="stone" />
        </div>
      </section>

      {chartData.length > 0 && (
        <section>
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">
            Top {chartData.length} features by PSI
          </h2>
          <div className="rounded-xl border hairline bg-white p-6">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="#E7E5E4" horizontal={false} />
                  <XAxis
                    type="number"
                    domain={[0, 'auto']}
                    tick={{ fill: '#78716C', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: '#78716C', fontSize: 10 }}
                    width={160}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    cursor={{ fill: '#F5F5F4' }}
                    contentStyle={{ borderRadius: 8, border: '1px solid #E7E5E4', fontSize: 12 }}
                    formatter={((v: number) => [v.toFixed(3), 'PSI']) as any}
                    labelFormatter={((_: unknown, payload: any) => payload?.[0]?.payload?.full || '') as any}
                  />
                  <ReferenceLine x={0.10} stroke="#F59E0B" strokeDasharray="3 3" />
                  <ReferenceLine x={0.25} stroke="#DC2626" strokeDasharray="3 3" />
                  <Bar dataKey="psi" radius={[0, 3, 3, 0]}>
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={
                        d.severity === 'significant' ? '#DC2626'
                        : d.severity === 'moderate' ? '#F59E0B'
                        : '#16A34A'
                      } />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-stone-500 mt-3">
              PSI &lt; 0.10 stable · 0.10–0.25 moderate · &gt; 0.25 significant (consider retraining).
            </p>
          </div>
        </section>
      )}

      {/* All features table */}
      <section>
        <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-4">All features</h2>
        <div className="rounded-xl border hairline bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-stone-50 text-xs uppercase tracking-wider text-stone-500">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium">Feature</th>
                <th className="px-4 py-2.5 text-right font-medium">PSI</th>
                <th className="px-4 py-2.5 text-left font-medium">Severity</th>
                <th className="px-4 py-2.5 text-right font-medium">N</th>
              </tr>
            </thead>
            <tbody className="divide-y hairline">
              {data.features.map(f => (
                <tr key={f.feature} className="hover:bg-stone-50">
                  <td className="px-4 py-2.5 font-numeric text-xs text-stone-700">{f.feature}</td>
                  <td className="px-4 py-2.5 text-right font-numeric text-stone-900">
                    {f.psi !== null ? f.psi.toFixed(3) : '—'}
                  </td>
                  <td className="px-4 py-2.5">
                    <SeverityChip severity={f.severity} />
                  </td>
                  <td className="px-4 py-2.5 text-right font-numeric text-stone-500 text-xs">
                    {f.n_current}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

function DriftTile({
  label, count, tone,
}: { label: string; count: number; tone: 'field' | 'harvest' | 'clay' | 'stone' }) {
  const cls = {
    field:   'text-field-700',
    harvest: 'text-harvest-700',
    clay:    'text-clay-700',
    stone:   'text-stone-700',
  }[tone]
  return (
    <div className="bg-white px-5 py-4">
      <p className="text-xs uppercase tracking-wider text-stone-500 font-medium">{label}</p>
      <p className={`mt-2 text-2xl font-semibold font-numeric ${cls}`}>{count}</p>
      <p className="mt-1 text-[11px] text-stone-500">features</p>
    </div>
  )
}

function SeverityChip({ severity }: { severity: string }) {
  if (severity === 'stable') {
    return <span className="text-xs font-medium text-field-700">stable</span>
  }
  if (severity === 'moderate') {
    return <span className="text-xs font-medium text-harvest-700">moderate</span>
  }
  if (severity === 'significant') {
    return <span className="text-xs font-medium text-clay-700">significant</span>
  }
  return <span className="text-xs text-stone-400">—</span>
}
