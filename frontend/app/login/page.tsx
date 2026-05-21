'use client'

import React, { Suspense, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'
import Alert from '@/components/ui/Alert'
import { Sprout, Mail, KeyRound, ArrowLeft } from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'
import { sendOTP, verifyOTP } from '@/lib/authApi'

type Step = 'email' | 'otp'

function LoginForm() {
  const router = useRouter()
  const search = useSearchParams()
  const redirectTo = search.get('redirect') || '/dashboard'

  const { setTokens, setUser } = useAuthStore()

  const [step, setStep] = useState<Step>('email')
  const [email, setEmail] = useState('')
  const [fullName, setFullName] = useState('')
  const [otp, setOtp] = useState('')
  const [otpExpiresInMin, setOtpExpiresInMin] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')

  const isValidEmail = (e: string) => /^[\w.+-]+@[\w-]+\.[\w.-]+$/.test(e)
  const isValidOtp = (o: string) => /^\d{6}$/.test(o)

  const handleSendOtp = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setInfo('')

    if (!isValidEmail(email)) {
      setError('Enter a valid email address')
      return
    }

    setLoading(true)
    try {
      const res = await sendOTP(email)
      setOtpExpiresInMin(res.expires_in_minutes ?? 10)
      setStep('otp')
      if (res.demo_otp) {
        // Demo mode — auto-fill the code so the recruiter can just click Verify
        setOtp(res.demo_otp)
        setInfo(`Demo mode — your code is ${res.demo_otp} (auto-filled below).`)
      } else {
        setInfo('Check your inbox for the 6-digit code.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send OTP')
    } finally {
      setLoading(false)
    }
  }

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!isValidOtp(otp)) {
      setError('OTP must be 6 digits')
      return
    }

    setLoading(true)
    try {
      const tokens = await verifyOTP(email, otp, fullName || undefined)
      setTokens(tokens.access_token, tokens.refresh_token)
      setUser({
        user_id: tokens.user_id,
        email: tokens.email,
        full_name: fullName || undefined,
        kyc_verified: false,
        is_active: true,
        created_at: new Date().toISOString(),
      })
      router.push(redirectTo)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to verify OTP')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-field-50 via-harvest-50/30 to-stone-50 flex items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2 mb-4">
            <Sprout className="w-10 h-10 text-field-600" />
            <span className="text-2xl font-bold text-stone-900">KisanCredit</span>
          </Link>
          <h1 className="text-3xl font-bold text-stone-900 mb-2">
            {step === 'email' ? 'Sign in / Get started' : 'Enter your code'}
          </h1>
          <p className="text-stone-600">
            {step === 'email'
              ? 'Passwordless sign-in — we email you a one-time code.'
              : `Sent to ${email}. Expires in ${otpExpiresInMin ?? 10} min.`}
          </p>
        </div>

        <Card>
          <AnimatePresence mode="wait">
            {step === 'email' ? (
              <motion.form
                key="email-step"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
                onSubmit={handleSendOtp}
                className="space-y-4"
              >
                {error && <Alert variant="error" message={error} />}

                <Input
                  label="Full name (only for new users)"
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Optional for existing users"
                  autoComplete="name"
                />

                <Input
                  label="Email address"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  icon={<Mail className="w-5 h-5" />}
                  autoComplete="email"
                  required
                />

                <Button type="submit" variant="primary" fullWidth loading={loading}>
                  Email me a code
                </Button>
              </motion.form>
            ) : (
              <motion.form
                key="otp-step"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                onSubmit={handleVerifyOtp}
                className="space-y-4"
              >
                {error && <Alert variant="error" message={error} />}
                {info && <Alert variant="info" message={info} />}

                <Input
                  label="6-digit code"
                  type="text"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  placeholder="------"
                  icon={<KeyRound className="w-5 h-5" />}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  required
                />

                <Button type="submit" variant="primary" fullWidth loading={loading}>
                  Verify &amp; continue
                </Button>

                <Button
                  type="button"
                  variant="ghost"
                  fullWidth
                  onClick={() => {
                    setStep('email')
                    setOtp('')
                    setError('')
                  }}
                  icon={<ArrowLeft className="w-4 h-4" />}
                >
                  Use a different email
                </Button>
              </motion.form>
            )}
          </AnimatePresence>
        </Card>
      </motion.div>
    </div>
  )
}

// useSearchParams() must sit inside a Suspense boundary or Next 16 bails out
// of static generation for this route. The fallback renders for the brief
// moment before the client hydrates the query string.
export default function LoginPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-stone-50" />}>
      <LoginForm />
    </Suspense>
  )
}
