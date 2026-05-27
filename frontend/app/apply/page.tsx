'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/lib/authStore'
import { ArrowRight, ArrowLeft, CheckCircle2, Loader2 } from 'lucide-react'
import { submitApplication, type LoanApplicationData } from '@/lib/api'
import { Button, Input, Alert } from '@/components/ui'

type FormStep = 'personal' | 'location' | 'loan' | 'financial' | 'employment'
type FlowState = 'form' | 'loading'

const STEPS = [
  { id: 'personal',   title: 'Personal'   },
  { id: 'location',   title: 'Location'   },
  { id: 'loan',       title: 'Loan'       },
  { id: 'financial',  title: 'Financial'  },
  { id: 'employment', title: 'Employment' },
] as const

// Home Credit category values — passed through to the backend mapping layer.
const EDUCATION_OPTIONS = [
  'Secondary / secondary special',
  'Higher education',
  'Incomplete higher',
  'Lower secondary',
  'Academic degree',
]
const HOUSING_OPTIONS = [
  'House / apartment',
  'Rented apartment',
  'With parents',
  'Municipal apartment',
  'Office apartment',
  'Co-op apartment',
]
const EMPLOYMENT_TYPE_OPTIONS = [
  'Working',
  'Commercial associate',
  'State servant',
  'Pensioner',
  'Businessman',
]

export default function ApplyPage() {
  const router = useRouter()
  const { isAuthenticated } = useAuthStore()
  const [flowState, setFlowState] = useState<FlowState>('form')
  const [currentStep, setCurrentStep] = useState<FormStep>('personal')
  const [formData, setFormData] = useState<LoanApplicationData>({
    fullName: '',
    mobile: '',
    dob: '',
    gender: '',
    occupation: '',
    pincode: '',
    loanAmount: 0,
    loanPurpose: '',
    monthlyIncome: 0,
    monthlyExpenses: 0,
    employmentYears: 0,
    employmentType: 'Working',
    educationLevel: 'Secondary / secondary special',
    housingType: 'House / apartment',
    dependents: 0,
    ownsCar: false,
    ownsProperty: false,
  })
  const [error, setError] = useState<string>('')

  const currentStepIndex = STEPS.findIndex(s => s.id === currentStep)
  const progress = ((currentStepIndex + 1) / STEPS.length) * 100

  useEffect(() => {
    if (!isAuthenticated) router.push('/login?redirect=/apply')
  }, [isAuthenticated, router])

  const NUMERIC_FIELDS = ['loanAmount', 'monthlyIncome', 'monthlyExpenses', 'employmentYears', 'dependents']

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const target = e.target
    const { name, value } = target
    let next: string | number | boolean = value
    if (target instanceof HTMLInputElement && target.type === 'checkbox') {
      next = target.checked
    } else if (NUMERIC_FIELDS.includes(name)) {
      next = parseFloat(value) || 0
    }
    setFormData(prev => ({ ...prev, [name]: next }))
  }

  const handleNext = () => {
    if (currentStepIndex < STEPS.length - 1) setCurrentStep(STEPS[currentStepIndex + 1].id as FormStep)
  }
  const handleBack = () => {
    if (currentStepIndex > 0) setCurrentStep(STEPS[currentStepIndex - 1].id as FormStep)
  }

  const handleSubmit = async () => {
    if (!isAuthenticated) { router.push('/login?redirect=/apply'); return }
    setError('')
    setFlowState('loading')
    try {
      const submitted = await submitApplication(formData)
      router.push(`/dashboard/applications/${submitted.application_id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process application')
      setFlowState('form')
    }
  }

  if (!isAuthenticated) return null

  if (flowState === 'loading') {
    return (
      <main className="min-h-screen bg-stone-50 flex items-center justify-center px-6">
        <div className="max-w-sm text-center">
          <Loader2 className="w-6 h-6 text-stone-400 mx-auto mb-4 animate-spin" />
          <p className="text-stone-900 font-medium">Submitting application…</p>
          <p className="mt-2 text-sm text-stone-500 leading-relaxed">
            The decision lifecycle takes ~15 seconds. You'll land on the live
            timeline page next. On a cold start the first request can take up
            to ~50s while the free-tier API wakes up.
          </p>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-stone-50 pt-24 pb-16 px-6">
      <div className="max-w-2xl mx-auto">
        <p className="text-sm text-stone-500 mb-2">New application</p>
        <h1 className="text-2xl font-semibold text-stone-900 tracking-tight">Loan application</h1>
        <p className="mt-2 text-sm text-stone-600">
          Five short sections. Fields map directly to the model's feature schema —
          no derived metrics are computed in the browser.
        </p>

        {/* Step indicator */}
        <div className="mt-8">
          <div className="flex items-center justify-between mb-3">
            {STEPS.map((step, i) => (
              <div key={step.id} className="flex-1 flex flex-col items-center">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold transition-colors ${
                  i < currentStepIndex
                    ? 'bg-stone-900 text-white'
                    : i === currentStepIndex
                    ? 'bg-harvest-500 text-white'
                    : 'bg-stone-200 text-stone-500'
                }`}>
                  {i < currentStepIndex ? <CheckCircle2 className="w-4 h-4" /> : i + 1}
                </div>
                <span className={`mt-1.5 text-[11px] font-medium ${
                  i === currentStepIndex ? 'text-stone-900' : 'text-stone-500'
                }`}>{step.title}</span>
              </div>
            ))}
          </div>
          <div className="h-0.5 bg-stone-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-harvest-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {error && (
          <div className="mt-6">
            <Alert variant="error" title="Error" onClose={() => setError('')}>{error}</Alert>
          </div>
        )}

        <div className="mt-8 rounded-xl border hairline bg-white p-8 shadow-soft">
          {currentStep === 'personal' && (
            <div className="space-y-5">
              <h2 className="text-base font-semibold text-stone-900">Personal information</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <Input label="Full name" name="fullName" value={formData.fullName} onChange={handleInputChange} placeholder="As on government ID" required />
                <Input label="Mobile" type="tel" name="mobile" value={formData.mobile} onChange={handleInputChange} placeholder="10 digits, starts 6-9" pattern="[0-9]{10}" required />
                <Input label="Date of birth" type="date" name="dob" value={formData.dob} onChange={handleInputChange} required />
                <div>
                  <label className="label label-required">Gender</label>
                  <select name="gender" value={formData.gender} onChange={handleInputChange} required className="input">
                    <option value="">Select</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {currentStep === 'location' && (
            <div className="space-y-5">
              <h2 className="text-base font-semibold text-stone-900">Location and occupation</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <Input label="Pincode" name="pincode" value={formData.pincode} onChange={handleInputChange} placeholder="6 digits" pattern="[0-9]{6}" required />
                <div>
                  <label className="label label-required">Occupation</label>
                  <select name="occupation" value={formData.occupation} onChange={handleInputChange} required className="input">
                    <option value="">Select</option>
                    <option value="farmer">Farmer</option>
                    <option value="self_employed">Self-employed</option>
                    <option value="salaried">Salaried</option>
                    <option value="business">Business owner</option>
                    <option value="daily_wage">Daily-wage worker</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {currentStep === 'loan' && (
            <div className="space-y-5">
              <h2 className="text-base font-semibold text-stone-900">Loan details</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <Input
                  label="Loan amount (₹)"
                  type="number"
                  name="loanAmount"
                  value={formData.loanAmount || ''}
                  onChange={handleInputChange}
                  placeholder="50,000 to 50,00,000"
                  min="50000"
                  max="5000000"
                  step="10000"
                  helperText="Between ₹50,000 and ₹50,00,000"
                  required
                />
                <div>
                  <label className="label label-required">Loan purpose</label>
                  <select name="loanPurpose" value={formData.loanPurpose} onChange={handleInputChange} required className="input">
                    <option value="">Select</option>
                    <option value="agriculture">Agriculture / farming</option>
                    <option value="business">Business expansion</option>
                    <option value="education">Education</option>
                    <option value="medical">Medical</option>
                    <option value="home_improvement">Home improvement</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {currentStep === 'financial' && (
            <div className="space-y-5">
              <h2 className="text-base font-semibold text-stone-900">Financials</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <Input
                  label="Monthly income (₹)"
                  type="number"
                  name="monthlyIncome"
                  value={formData.monthlyIncome || ''}
                  onChange={handleInputChange}
                  placeholder="e.g. 45,000"
                  min="0"
                  step="1000"
                  helperText="Take-home, per month"
                  required
                />
                <Input
                  label="Monthly expenses (₹)"
                  type="number"
                  name="monthlyExpenses"
                  value={formData.monthlyExpenses || ''}
                  onChange={handleInputChange}
                  placeholder="e.g. 22,000"
                  min="0"
                  step="1000"
                  required
                />
              </div>
            </div>
          )}

          {currentStep === 'employment' && (
            <div className="space-y-5">
              <h2 className="text-base font-semibold text-stone-900">Employment and household</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <Input
                  label="Years in current employment"
                  type="number"
                  name="employmentYears"
                  value={formData.employmentYears || ''}
                  onChange={handleInputChange}
                  placeholder="e.g. 5"
                  min="0"
                  max="50"
                  step="0.5"
                  required
                />
                <div>
                  <label className="label label-required">Employment type</label>
                  <select name="employmentType" value={formData.employmentType} onChange={handleInputChange} required className="input">
                    {EMPLOYMENT_TYPE_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
                <div>
                  <label className="label label-required">Education</label>
                  <select name="educationLevel" value={formData.educationLevel} onChange={handleInputChange} required className="input">
                    {EDUCATION_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
                <div>
                  <label className="label label-required">Housing</label>
                  <select name="housingType" value={formData.housingType} onChange={handleInputChange} required className="input">
                    {HOUSING_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
                <Input
                  label="Dependents"
                  type="number"
                  name="dependents"
                  value={formData.dependents || ''}
                  onChange={handleInputChange}
                  placeholder="e.g. 2"
                  min="0"
                  max="15"
                  required
                />
              </div>

              <div className="flex flex-wrap gap-6 pt-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox" name="ownsCar"
                    checked={formData.ownsCar} onChange={handleInputChange}
                    className="w-4 h-4 text-harvest-600 border-stone-300 rounded focus:ring-harvest-500"
                  />
                  <span className="text-sm text-stone-700">I own a car</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox" name="ownsProperty"
                    checked={formData.ownsProperty} onChange={handleInputChange}
                    className="w-4 h-4 text-harvest-600 border-stone-300 rounded focus:ring-harvest-500"
                  />
                  <span className="text-sm text-stone-700">I own property</span>
                </label>
              </div>

              <p className="text-xs text-stone-500 leading-relaxed pt-2 border-t hairline">
                The decision and the SHAP factors behind it appear on the next screen.
              </p>
            </div>
          )}

          {/* Navigation */}
          <div className="flex gap-3 mt-8 pt-6 border-t hairline">
            {currentStepIndex > 0 && (
              <Button
                variant="secondary"
                size="md"
                onClick={handleBack}
                icon={<ArrowLeft className="w-4 h-4" />}
              >
                Back
              </Button>
            )}
            {currentStepIndex < STEPS.length - 1 ? (
              <Button
                variant="primary"
                size="md"
                fullWidth
                onClick={handleNext}
                icon={<ArrowRight className="w-4 h-4" />}
                iconPosition="right"
              >
                Continue
              </Button>
            ) : (
              <Button
                variant="primary"
                size="md"
                fullWidth
                onClick={handleSubmit}
                icon={<CheckCircle2 className="w-4 h-4" />}
                iconPosition="right"
              >
                Submit application
              </Button>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
