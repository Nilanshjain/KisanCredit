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
  XCircle,
  AlertTriangle,
  Smartphone,
  TrendingUp,
  Home,
  Sparkles
} from 'lucide-react'
import { generateFeaturesFromApplication, predictLoan, getDecision, calculateEMI, type LoanApplicationData, type PredictionResponse } from '@/lib/api'
import { Button, Input, Card, Alert, Skeleton } from '@/components/ui'

type FormStep = 'personal' | 'location' | 'loan' | 'financial' | 'digital'
type FlowState = 'form' | 'loading' | 'result'

const STEPS = [
  { id: 'personal', title: 'Personal Info', icon: User },
  { id: 'location', title: 'Location', icon: Briefcase },
  { id: 'loan', title: 'Loan Details', icon: IndianRupee },
  { id: 'financial', title: 'Financial', icon: FileText },
  { id: 'digital', title: 'Digital Data', icon: Smartphone },
] as const

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
    upiTransactions: 50,
    totalContacts: 300,
    smsCount: 500,
    appUsageMinutes: 180,
  })
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string>('')

  const currentStepIndex = STEPS.findIndex(s => s.id === currentStep)
  const progress = ((currentStepIndex + 1) / STEPS.length) * 100

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login?redirect=/apply')
    }
  }, [isAuthenticated, router])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: ['loanAmount', 'monthlyIncome', 'monthlyExpenses', 'upiTransactions', 'totalContacts', 'smsCount', 'appUsageMinutes'].includes(name)
        ? parseFloat(value) || 0
        : value
    }))
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
      const features = generateFeaturesFromApplication(formData)
      const prediction = await predictLoan(features)
      setResult(prediction)
      setFlowState('result')
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
          <p className="text-lg text-stone-600 mb-8">Our AI is reviewing your details...</p>

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

  // Result State
  if (flowState === 'result' && result) {
    const decision = getDecision(result.credit_score)
    const emi = decision === 'APPROVED' ? calculateEMI(formData.loanAmount, 18, 12) : 0

    const ResultIcon = decision === 'APPROVED' ? CheckCircle2 : decision === 'MANUAL_REVIEW' ? AlertTriangle : XCircle
    const colorClasses = {
      APPROVED: { bg: 'from-field-400 to-field-600', text: 'text-field-700', card: 'border-field-200 bg-field-50' },
      MANUAL_REVIEW: { bg: 'from-harvest-400 to-harvest-600', text: 'text-harvest-700', card: 'border-harvest-200 bg-harvest-50' },
      REJECTED: { bg: 'from-clay-400 to-clay-600', text: 'text-clay-700', card: 'border-clay-200 bg-clay-50' },
    }
    const colors = colorClasses[decision as keyof typeof colorClasses]

    return (
      <div className="min-h-screen flex items-center justify-center p-4 bg-rural-pattern">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-2xl w-full"
        >
          <Card padding="lg">
            {/* Result Icon */}
            <div className="text-center mb-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 15 }}
                className={`w-24 h-24 mx-auto mb-6 bg-gradient-to-br ${colors.bg} rounded-full flex items-center justify-center shadow-soft-lg`}
              >
                <ResultIcon className="w-12 h-12 text-stone-900" />
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className={`text-3xl font-bold ${colors.text} mb-2`}
              >
                {decision === 'APPROVED' && 'Loan Approved!'}
                {decision === 'MANUAL_REVIEW' && 'Under Review'}
                {decision === 'REJECTED' && 'Not Approved'}
              </motion.h1>

              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="text-lg text-stone-600"
              >
                {decision === 'APPROVED' && `Congratulations ${formData.fullName}! Your loan has been approved.`}
                {decision === 'MANUAL_REVIEW' && `Thank you ${formData.fullName}. We'll review and respond within 24 hours.`}
                {decision === 'REJECTED' && `We're sorry ${formData.fullName}, we cannot approve your loan at this time.`}
              </motion.p>
            </div>

            {/* Approval Details */}
            {decision === 'APPROVED' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="space-y-6"
              >
                <div className="grid md:grid-cols-2 gap-4">
                  <Card className={colors.card}>
                    <div className="p-4">
                      <p className="text-sm text-field-600 mb-1">Loan Amount</p>
                      <p className="text-3xl font-bold text-field-700">₹{formData.loanAmount.toLocaleString('en-IN')}</p>
                    </div>
                  </Card>
                  <Card className={colors.card}>
                    <div className="p-4">
                      <p className="text-sm text-field-600 mb-1">Monthly EMI</p>
                      <p className="text-3xl font-bold text-field-700">₹{emi.toLocaleString('en-IN')}</p>
                    </div>
                  </Card>
                  <Card className="border-harvest-200 bg-harvest-50">
                    <div className="p-4">
                      <p className="text-sm text-harvest-600 mb-1">Credit Score</p>
                      <p className="text-3xl font-bold text-harvest-700">{Math.round(result.credit_score)}/100</p>
                    </div>
                  </Card>
                  <Card className="border-harvest-200 bg-harvest-50">
                    <div className="p-4">
                      <p className="text-sm text-harvest-600 mb-1">Processing Fee</p>
                      <p className="text-3xl font-bold text-harvest-700">₹8</p>
                    </div>
                  </Card>
                </div>

                <Alert variant="success" title="Top Approval Factors">
                  <ul className="space-y-2 mt-2">
                    {result.top_features.slice(0, 3).map((feature, idx) => (
                      <li key={idx} className="flex items-center gap-2 text-sm">
                        <TrendingUp className="w-4 h-4 text-field-600" />
                        <span>{feature.name}: <strong>{feature.value.toFixed(2)}</strong></span>
                      </li>
                    ))}
                  </ul>
                </Alert>
              </motion.div>
            )}

            {/* Manual Review Details */}
            {decision === 'MANUAL_REVIEW' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Alert variant="warning">
                  <div className="space-y-2">
                    <p><strong>Loan Amount:</strong> ₹{formData.loanAmount.toLocaleString('en-IN')}</p>
                    <p><strong>Credit Score:</strong> {Math.round(result.credit_score)}/100</p>
                    <p><strong>Reference ID:</strong> KC{Date.now().toString().slice(-8)}</p>
                  </div>
                </Alert>
              </motion.div>
            )}

            {/* Rejection Details */}
            {decision === 'REJECTED' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="space-y-4"
              >
                <Alert variant="error" title="Reasons">
                  <ul className="space-y-2 mt-2">
                    <li className="flex items-start gap-2 text-sm">
                      <XCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>Credit score below threshold ({Math.round(result.credit_score)}/100)</span>
                    </li>
                    <li className="flex items-start gap-2 text-sm">
                      <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>You can reapply after 30 days</span>
                    </li>
                  </ul>
                </Alert>

                <Alert variant="info" title="How to Improve">
                  <ul className="space-y-1 mt-2 text-sm">
                    <li>• Increase monthly UPI transactions</li>
                    <li>• Maintain regular income patterns</li>
                    <li>• Reduce expenses relative to income</li>
                    <li>• Build a stronger digital footprint</li>
                  </ul>
                </Alert>
              </motion.div>
            )}

            {/* Action Buttons */}
            <div className="mt-8 flex gap-4">
              <Button
                variant="secondary"
                size="lg"
                fullWidth
                onClick={() => router.push('/')}
                icon={<Home className="w-5 h-5" />}
              >
                Back to Home
              </Button>
              {decision === 'REJECTED' && (
                <Button
                  variant="ghost"
                  size="lg"
                  fullWidth
                  onClick={() => {
                    setFlowState('form')
                    setCurrentStep('personal')
                  }}
                >
                  Try Again
                </Button>
              )}
            </div>
          </Card>
        </motion.div>
      </div>
    )
  }

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
                    placeholder="₹10,000 - ₹1,00,000"
                    min="1000"
                    max="100000"
                    step="1000"
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
                    placeholder="Your monthly income"
                    min="0"
                    step="1000"
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

            {/* Step 5: Digital Activity */}
            {currentStep === 'digital' && (
              <motion.div
                key="digital"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                <div>
                  <h2 className="text-2xl font-bold text-stone-900 flex items-center gap-2 mb-2">
                    <Smartphone className="w-6 h-6 text-harvest-600" />
                    Digital Activity
                  </h2>
                  <p className="text-sm text-stone-600">Optional - helps improve approval chances</p>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <Input
                    label="UPI Transactions (per month)"
                    type="number"
                    name="upiTransactions"
                    value={formData.upiTransactions || ''}
                    onChange={handleInputChange}
                    placeholder="Approx. UPI transactions"
                    min="0"
                  />
                  <Input
                    label="Total Contacts"
                    type="number"
                    name="totalContacts"
                    value={formData.totalContacts || ''}
                    onChange={handleInputChange}
                    placeholder="Contacts in phone"
                    min="0"
                  />
                  <Input
                    label="SMS Count (per month)"
                    type="number"
                    name="smsCount"
                    value={formData.smsCount || ''}
                    onChange={handleInputChange}
                    placeholder="Approx. SMS received"
                    min="0"
                  />
                  <Input
                    label="App Usage (minutes/day)"
                    type="number"
                    name="appUsageMinutes"
                    value={formData.appUsageMinutes || ''}
                    onChange={handleInputChange}
                    placeholder="Daily app usage"
                    min="0"
                  />
                </div>

                <Alert variant="info">
                  <p className="text-sm">
                    <strong>Note:</strong> Your data is encrypted and secure. We only use it for loan assessment. Processing fee of ₹8 applies on approval.
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
