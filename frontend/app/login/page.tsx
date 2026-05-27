'use client'

import React, { Suspense, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Alert from '@/components/ui/Alert'
import { Mail, KeyRound, ArrowLeft } from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'
import { sendOTP, verifyOTP, getCurrentUser } from '@/lib/authApi'

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
        setOtp(res.demo_otp)
        setInfo(`Demo mode — code ${res.demo_otp} auto-filled.`)
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

      let role: 'user' | 'admin' = 'user'
      try {
        const profile = await getCurrentUser()
        setUser(profile)
        role = profile.role === 'admin' ? 'admin' : 'user'
      } catch {
        setUser({
          user_id: tokens.user_id,
          email: tokens.email,
          full_name: fullName || undefined,
          kyc_verified: false,
          is_active: true,
          created_at: new Date().toISOString(),
        })
      }

      const dest = role === 'admin' && redirectTo === '/dashboard' ? '/admin' : redirectTo
      router.push(dest)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to verify OTP')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-stone-50 flex items-center justify-center px-6 py-16">
      <div className="w-full max-w-sm">
        <div className="mb-10">
          <Link href="/" className="text-sm text-stone-500 hover:text-stone-900 inline-flex items-center gap-1.5 mb-8">
            <ArrowLeft className="w-3.5 h-3.5" /> Back
          </Link>
          <h1 className="text-2xl font-semibold text-stone-900 tracking-tight">
            {step === 'email' ? 'Sign in' : 'Enter the code'}
          </h1>
          <p className="mt-2 text-sm text-stone-600">
            {step === 'email'
              ? 'Email-OTP. New emails create an account automatically.'
              : `Sent to ${email} · expires in ${otpExpiresInMin ?? 10} min.`}
          </p>
        </div>

        <div className="rounded-xl border hairline bg-white p-6 shadow-soft">
          {step === 'email' ? (
            <form onSubmit={handleSendOtp} className="space-y-4">
              {error && <Alert variant="error" message={error} />}

              <Input
                label="Full name"
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Optional — only used the first time"
                autoComplete="name"
              />

              <Input
                label="Email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                icon={<Mail className="w-4 h-4" />}
                autoComplete="email"
                required
              />

              <Button type="submit" variant="primary" fullWidth loading={loading}>
                Send code
              </Button>
            </form>
          ) : (
            <form onSubmit={handleVerifyOtp} className="space-y-4">
              {error && <Alert variant="error" message={error} />}
              {info && <Alert variant="info" message={info} />}

              <Input
                label="6-digit code"
                type="text"
                value={otp}
                onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                placeholder="------"
                icon={<KeyRound className="w-4 h-4" />}
                inputMode="numeric"
                autoComplete="one-time-code"
                required
              />

              <Button type="submit" variant="primary" fullWidth loading={loading}>
                Verify and continue
              </Button>

              <button
                type="button"
                onClick={() => { setStep('email'); setOtp(''); setError('') }}
                className="w-full text-sm text-stone-500 hover:text-stone-900 py-1"
              >
                Use a different email
              </button>
            </form>
          )}
        </div>
      </div>
    </main>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-stone-50" />}>
      <LoginForm />
    </Suspense>
  )
}
