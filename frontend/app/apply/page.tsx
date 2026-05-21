'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import {
  Sprout,
  IndianRupee,
  User,
  Phone,
  Briefcase,
  FileText,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  Building2,
  Sparkles,
} from 'lucide-react'
import { submitApplication, type LoanApplicationData } from '@/lib/api'
import { Button, Input, Card, Alert, Skeleton } from '@/components/ui'

type FormStep = 'personal' | 'location' | 'loan' | 'financial' | 'employment'
type FlowState = 'form' | 'loading'   // 'result' lives on /dashboard/applications/[id] now

const STEPS = [
  { id: 'personal', title: 'Personal', icon: User },
  { id: 'location', title: 'Location', icon: Briefcase },
  { id: 'loan', title: 'Loan', icon: IndianRupee },
  { id: 'financial', title: 'Financial', icon: FileText },
  { id: 'employment', title: 'Employment', icon: Building2 },
] as const

// Home Credit category values — sent as-is so the backend maps them directly.
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

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login?redirect=/apply')
    }
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
    if (currentStepIndex < STEPS.length - 1) {
      setCurrentStep(STEPS[currentStepIndex + 1].id as FormStep)
    }
  }

  const handleBack = () => {
    if (currentStepIndex > 0) {
      setCurrentStep(STEPS[currentStepIndex - 1].id as FormStep)
    }
  }

  const handleSubmit = async () => {
    if (!isAuthenticated) {
      router.push('/login?redirect=/apply')
      return
    }

    setError('')
    setFlowState('loading')

    try {
      // Backend kicks off an async lifecycle (submitted -> under_review -> decided).
      // Hand off to the detail page where the timeline is rendered live.
      const submitted = await submitApplication(formData)
      router.push(`/dashboard/applications/${submitted.application_id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process application')
      setFlowState('form')
    }
  }

  // Show nothing while checking authentication
  if (!isAuthenticated) {
    return null
  }

  // Loading State
  if (flowState === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-rural-pattern">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center max-w-md"
        >
          <div className="mb-8">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="w-24 h-24 mx-auto bg-gradient-harvest rounded-full flex items-center justify-center shadow-soft-lg"
            >
              <Sparkles className="w-12 h-12 text-stone-900" />
            </motion.div>
          </div>

          <h2 className="text-3xl font-bold text-stone-900 mb-4">Analyzing Your Application</h2>
          <p className="text-lg text-stone-600 mb-4">Our AI is reviewing your details...</p>
          <p className="text-sm text-stone-500 mb-8">
            First request may take ~50 seconds while the model warms up (free-tier infra).
          </p>

          <div className="space-y-4">
            <Skeleton variant="text" className="w-full h-4" />
            <Skeleton variant="text" className="w-3/4 h-4 mx-auto" />
            <Skeleton variant="text" className="w-5/6 h-4 mx-auto" />
          </div>

          <div className="mt-8 flex justify-center gap-2">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="w-3 h-3 bg-harvest-500 rounded-full"
                animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
              />
            ))}
          </div>
        </motion.div>
      </div>
    )
  }

  // Result state lives on /dashboard/applications/[id] — handleSubmit redirects there
  // once the application is enqueued, and that page renders the live timeline
  // (submitted -> under_review -> decided) plus the decision + SHAP breakdown.

  // Form State (Multi-step wizard)
  return (
    <div className="min-h-screen bg-rural-pattern py-8 px-4 relative">
      {/* Decorative background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-32 h-32 opacity-5">
          <svg viewBox="0 0 100 100" fill="currentColor" className="text-field-600">
            <path d="M50 10 L50 90 M45 20 Q40 25 45 30 M55 20 Q60 25 55 30 M45 35 Q40 40 45 45 M55 35 Q60 40 55 45 M45 50 Q40 55 45 60 M55 50 Q60 55 55 60 M45 65 Q40 70 45 75 M55 65 Q60 70 55 75" stroke="currentColor" strokeWidth="2" fill="none"/>
          </svg>
        </div>
        <div className="absolute bottom-20 right-20 w-40 h-40 opacity-5">
          <svg viewBox="0 0 100 100" fill="currentColor" className="text-harvest-500">
            <path d="M50 10 L50 90 M45 20 Q40 25 45 30 M55 20 Q60 25 55 30 M45 35 Q40 40 45 45 M55 35 Q60 40 55 45 M45 50 Q40 55 45 60 M55 50 Q60 55 55 60 M45 65 Q40 70 45 75 M55 65 Q60 70 55 75" stroke="currentColor" strokeWidth="2" fill="none"/>
          </svg>
        </div>
      </div>
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Sprout className="w-10 h-10 text-harvest-500" />
            <h1 className="text-4xl font-bold text-stone-900">Apply for Loan</h1>
          </div>
          <p className="text-lg text-stone-600">Get instant approval in under 60 seconds</p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {STEPS.map((step, index) => {
              const isActive = step.id === currentStep
              const isCompleted = index < currentStepIndex
              const StepIcon = step.icon

              return (
                <div key={step.id} className="flex items-center flex-1">
                  <div className="flex flex-col items-center flex-1">
                    <motion.div
                      className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                        isActive ? 'border-harvest-500 bg-harvest-50 text-harvest-600' :
                        isCompleted ? 'border-field-500 bg-field-500 text-stone-900' :
                        'border-stone-300 bg-white text-stone-400'
                      }`}
                      whileHover={{ scale: 1.05 }}
                    >
                      {isCompleted ? <CheckCircle2 className="w-6 h-6" /> : <StepIcon className="w-6 h-6" />}
                    </motion.div>
                    <span className={`text-xs mt-2 font-medium ${isActive ? 'text-harvest-600' : 'text-stone-500'}`}>
                      {step.title}
                    </span>
                  </div>
                  {index < STEPS.length - 1 && (
                    <div className={`h-0.5 flex-1 mx-2 transition-colors ${isCompleted ? 'bg-field-500' : 'bg-stone-200'}`} />
                  )}
                </div>
              )
            })}
          </div>

          <div className="h-2 bg-stone-200 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-harvest"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* Error Alert */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6"
            >
              <Alert variant="error" title="Error" onClose={() => setError('')}>
                {error}
              </Alert>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Form Card */}
        <Card padding="lg">
          <AnimatePresence mode="wait">
            {/* Step 1: Personal Information */}
            {currentStep === 'personal' && (
              <motion.div
                key="personal"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <h2 className="text-2xl font-bold text-stone-900 flex items-center gap-2">
                  <User className="w-6 h-6 text-harvest-600" />
                  Personal Information
                </h2>

                <div className="grid md:grid-cols-2 gap-4">
                  <Input
                    label="Full Name"
                    name="fullName"
                    value={formData.fullName}
                    onChange={handleInputChange}
                    placeholder="Enter your full name"
                    required
                  />
                  <Input
                    label="Mobile Number"
                    type="tel"
                    name="mobile"
                    value={formData.mobile}
                    onChange={handleInputChange}
                    placeholder="10-digit mobile"
                    pattern="[0-9]{10}"
                    icon={<Phone className="w-5 h-5" />}
                    required
                  />
                  <Input
                    label="Date of Birth"
                    type="date"
                    name="dob"
                    value={formData.dob}
                    onChange={handleInputChange}
                    required
                  />
                  <div>
                    <label className="label label-required">Gender</label>
                    <select name="gender" value={formData.gender} onChange={handleInputChange} required className="input">
                      <option value="">Select gender</option>
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 2: Location */}
            {currentStep === 'location' && (
              <motion.div
                key="location"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <h2 className="text-2xl font-bold text-stone-900 flex items-center gap-2">
                  <Briefcase className="w-6 h-6 text-harvest-600" />
                  Location & Occupation
                </h2>

                <div className="grid md:grid-cols-2 gap-4">
                  <Input
                    label="Pincode"
                    name="pincode"
                    value={formData.pincode}
                    onChange={handleInputChange}
                    placeholder="6-digit pincode"
                    pattern="[0-9]{6}"
                    required
                  />
                  <div>
                    <label className="label label-required">Occupation</label>
                    <select name="occupation" value={formData.occupation} onChange={handleInputChange} required className="input">
                      <option value="">Select occupation</option>
                      <option value="farmer">Farmer</option>
                      <option value="self_employed">Self Employed</option>
                      <option value="salaried">Salaried</option>
                      <option value="business">Business Owner</option>
                      <option value="daily_wage">Daily Wage Worker</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 3: Loan Details */}
            {currentStep === 'loan' && (
              <motion.div
                key="loan"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <h2 className="text-2xl font-bold text-stone-900 flex items-center gap-2">
                  <IndianRupee className="w-6 h-6 text-harvest-600" />
                  Loan Details
                </h2>

                <div className="grid md:grid-cols-2 gap-4">
                  <Input
                    label="Loan Amount (₹)"
                    type="number"
                    name="loanAmount"
                    value={formData.loanAmount || ''}
                    onChange={handleInputChange}
                    placeholder="₹3,00,000 - ₹30,00,000"
                    min="50000"
                    max="5000000"
                    step="50000"
                    helperText="Typical range ₹3-30 lakh"
                    icon={<IndianRupee className="w-5 h-5" />}
                    required
                  />
                  <div>
                    <label className="label label-required">Loan Purpose</label>
                    <select name="loanPurpose" value={formData.loanPurpose} onChange={handleInputChange} required className="input">
                      <option value="">Select purpose</option>
                      <option value="agriculture">Agriculture/Farming</option>
                      <option value="business">Business Expansion</option>
                      <option value="education">Education</option>
                      <option value="medical">Medical Emergency</option>
                      <option value="home_improvement">Home Improvement</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 4: Financial Information */}
            {currentStep === 'financial' && (
              <motion.div
                key="financial"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <h2 className="text-2xl font-bold text-stone-900 flex items-center gap-2">
                  <FileText className="w-6 h-6 text-harvest-600" />
                  Financial Information
                </h2>

                <div className="grid md:grid-cols-2 gap-4">
                  <Input
                    label="Monthly Income (₹)"
                    type="number"
                    name="monthlyIncome"
                    value={formData.monthlyIncome || ''}
                    onChange={handleInputChange}
                    placeholder="e.g. 45,000"
                    min="0"
                    step="1000"
                    helperText="Take-home pay per month"
                    icon={<IndianRupee className="w-5 h-5" />}
                    required
                  />
                  <Input
                    label="Monthly Expenses (₹)"
                    type="number"
                    name="monthlyExpenses"
                    value={formData.monthlyExpenses || ''}
                    onChange={handleInputChange}
                    placeholder="Your monthly expenses"
                    min="0"
                    step="1000"
                    icon={<IndianRupee className="w-5 h-5" />}
                    required
                  />
                </div>
              </motion.div>
            )}

            {/* Step 5: Employment & Household */}
            {currentStep === 'employment' && (
              <motion.div
                key="employment"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <div>
                  <h2 className="text-2xl font-bold text-stone-900 flex items-center gap-2 mb-2">
                    <Building2 className="w-6 h-6 text-harvest-600" />
                    Employment &amp; Household
                  </h2>
                  <p className="text-sm text-stone-600">These factors directly shape your credit assessment.</p>
                </div>

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
                    label="Number of dependents"
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

                <div className="flex flex-wrap gap-6">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      name="ownsCar"
                      checked={formData.ownsCar}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-harvest-600 border-stone-300 rounded focus:ring-harvest-500"
                    />
                    <span className="text-sm text-stone-700">I own a car</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      name="ownsProperty"
                      checked={formData.ownsProperty}
                      onChange={handleInputChange}
                      className="w-4 h-4 text-harvest-600 border-stone-300 rounded focus:ring-harvest-500"
                    />
                    <span className="text-sm text-stone-700">I own property / real estate</span>
                  </label>
                </div>

                <Alert variant="info">
                  <p className="text-sm">
                    Your details are used only for this credit assessment. The decision and an
                    explanation of the factors behind it appear on the next screen.
                  </p>
                </Alert>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Navigation Buttons */}
          <div className="flex gap-4 mt-8">
            {currentStepIndex > 0 && (
              <Button
                variant="secondary"
                size="lg"
                onClick={handleBack}
                icon={<ArrowLeft className="w-5 h-5" />}
              >
                Back
              </Button>
            )}

            {currentStepIndex < STEPS.length - 1 ? (
              <Button
                variant="primary"
                size="lg"
                fullWidth
                onClick={handleNext}
                icon={<ArrowRight className="w-5 h-5" />}
                iconPosition="right"
              >
                Continue
              </Button>
            ) : (
              <Button
                variant="success"
                size="lg"
                fullWidth
                onClick={handleSubmit}
                icon={<CheckCircle2 className="w-5 h-5" />}
                iconPosition="right"
              >
                Submit Application
              </Button>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
