'use client'

import { useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, Github } from 'lucide-react'
import { Button } from '@/components/ui'

// Numbers are mirrored from the training artifact and README.
// Update them only when retrain results land.
const MODEL_FACTS = [
  { label: 'ROC-AUC (LightGBM, 5-fold CV)', value: '0.6803 ± 0.0047' },
  { label: 'Applicants', value: '307,511' },
  { label: 'Features (application-time only)', value: '39' },
  { label: 'Logistic-regression baseline', value: '0.6578' },
  { label: 'TabNet baseline (PyTorch)', value: '0.6524' },
  { label: 'Default rate (class balance)', value: '8.07%' },
]

const SYSTEM_PIECES = [
  {
    heading: 'Borrower flow',
    body: 'Email-OTP sign-in, a multi-step application form, and a lifecycle state machine (submitted → under_review → decided) that runs as a FastAPI background task. Each transition is written as an immutable audit event the frontend polls until the decision lands.',
  },
  {
    heading: 'Explanations',
    body: 'SHAP TreeExplainer surfaces the top contributors behind every decision. Gemini Flash narrates them in English or Hindi. When the answer isn\'t yes, a greedy counter-factual search proposes the smallest realistic change that would move the decision toward approve.',
  },
  {
    heading: 'Lender operations console',
    body: 'A separate /admin surface for operators: live metrics with model performance and human-override activity tracked separately, a paginated application queue with full drill-down, audit-trailed manual overrides, and per-feature PSI drift monitoring against the training baseline.',
  },
  {
    heading: 'Decision thresholds, not cutoffs',
    body: 'On an 8%-default population a fixed cutoff like "approve above 0.6" approves everyone. The approve / manual_review / reject bands are derived from percentiles of the model\'s own score distribution on the training set and stored inside the model artifact — making the policy explicit and auditable.',
  },
]

export default function HomePage() {
  // Pre-warm Render's free-tier API (~50s cold start after 15min idle).
  // Best-effort; never blocks UI.
  useEffect(() => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api/v1'
    fetch(`${apiBase}/health`, { method: 'GET', cache: 'no-store' }).catch(() => {})
  }, [])

  return (
    <main className="min-h-screen bg-stone-50">
      {/* Hero */}
      <section className="pt-32 pb-20">
        <div className="max-w-3xl mx-auto px-6">
          <p className="text-sm font-medium text-stone-500 mb-4">
            KisanCredit · ML credit-scoring system
          </p>
          <h1 className="text-4xl md:text-5xl font-semibold text-stone-900 tracking-tight leading-[1.1]">
            Credit scoring for thin-file borrowers.
          </h1>
          <p className="mt-6 text-lg text-stone-600 leading-relaxed">
            A LightGBM default-risk model trained on 307,511 real loan applicants
            from the Home Credit dataset, served through a production stack
            with a lender operations console, SHAP explanations, and Gemini-generated
            natural-language reasoning in English and Hindi.
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-3">
            <Link href="/login">
              <Button variant="primary" size="md" icon={<ArrowRight className="w-4 h-4" />} iconPosition="right">
                Try the demo
              </Button>
            </Link>
            <a
              href="https://github.com/Nilanshjain/KisanCredit"
              target="_blank"
              rel="noreferrer"
            >
              <Button variant="secondary" size="md" icon={<Github className="w-4 h-4" />}>
                Source on GitHub
              </Button>
            </a>
          </div>
        </div>
      </section>

      {/* Validation results */}
      <section className="border-t hairline bg-white">
        <div className="max-w-3xl mx-auto px-6 py-16">
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-6">
            Validation
          </h2>
          <dl className="font-numeric grid sm:grid-cols-2 gap-x-12 gap-y-5">
            {MODEL_FACTS.map(f => (
              <div key={f.label} className="flex items-baseline justify-between border-b hairline pb-3">
                <dt className="text-sm text-stone-600">{f.label}</dt>
                <dd className="text-base font-semibold text-stone-900">{f.value}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-8 text-sm text-stone-500 leading-relaxed">
            The Kaggle leaderboard for Home Credit tops out near 0.805 — that
            number uses the full multi-table dataset (bureau records, prior
            applications, payment histories). This model is trained on
            application-only features so it works for borrowers who have no
            credit-bureau record. The ~8 AUC-point gap is the price of that
            choice.
          </p>
        </div>
      </section>

      {/* What's in the system */}
      <section className="border-t hairline">
        <div className="max-w-3xl mx-auto px-6 py-16">
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-6">
            What's in the system
          </h2>
          <div className="space-y-10">
            {SYSTEM_PIECES.map(p => (
              <div key={p.heading}>
                <h3 className="text-lg font-semibold text-stone-900 mb-2">{p.heading}</h3>
                <p className="text-stone-600 leading-relaxed">{p.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Honest about limitations */}
      <section className="border-t hairline bg-white">
        <div className="max-w-3xl mx-auto px-6 py-16">
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500 mb-6">
            Honest about what this isn't
          </h2>
          <ul className="space-y-4 text-stone-600 leading-relaxed">
            <li>
              <span className="font-medium text-stone-900">No Indian dataset.</span>{' '}
              Individual-level Indian credit data isn't publicly available. Home
              Credit is a real emerging-market thin-file proxy, but a production
              Indian deployment would need CIBIL access or a partner-NBFC data agreement.
            </li>
            <li>
              <span className="font-medium text-stone-900">Modest accuracy, by design.</span>{' '}
              AUC 0.68 is well below the ~0.80 reachable with bureau features.
              That gap is the price of not using a bureau score for the thin-file
              population. It's a real, if imperfect, ranking of risk — not a
              finished underwriting model.
            </li>
            <li>
              <span className="font-medium text-stone-900">Free-tier hosting.</span>{' '}
              Render sleeps the API after 15 minutes idle and takes roughly 50
              seconds to wake. The frontend shows a warming-up state to cover it.
            </li>
          </ul>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t hairline">
        <div className="max-w-3xl mx-auto px-6 py-16 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
          <div>
            <h2 className="text-2xl font-semibold text-stone-900">Try the demo</h2>
            <p className="mt-2 text-stone-600">
              Email-OTP login. In demo mode the code is auto-filled, so no inbox is needed.
            </p>
          </div>
          <Link href="/login">
            <Button variant="primary" size="lg" icon={<ArrowRight className="w-4 h-4" />} iconPosition="right">
              Try the demo
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t hairline">
        <div className="max-w-3xl mx-auto px-6 py-10 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 text-sm text-stone-500">
          <span>KisanCredit · Nilansh Jain · MIT License</span>
          <div className="flex gap-6">
            <Link href="/terms" className="hover:text-stone-900">Terms</Link>
            <Link href="/privacy" className="hover:text-stone-900">Privacy</Link>
            <Link href="/contact" className="hover:text-stone-900">Contact</Link>
          </div>
        </div>
      </footer>
    </main>
  )
}
