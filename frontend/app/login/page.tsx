'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'
import Alert from '@/components/ui/Alert'
import { Sprout, Phone, KeyRound, ArrowLeft } from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'
import { sendOTP, verifyOTP } from '@/lib/authApi'

type Step = 'phone' | 'otp'

export default function LoginPage() {
  const router = useRouter()
  const search = useSearchParams()
  const redirectTo = search.get('redirect') || '/dashboard'

  const { setTokens, setUser } = useAuthStore()

  const [step, setStep] = useState<Step>('phone')
  const [phone, setPhone] = useState('')
  const [fullName, setFullName] = useState('')
  const [otp, setOtp] = useState('')
  const [otpExpiresInMin, setOtpExpiresInMin] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')

  const isValidPhone = (p: string) => /^[6-9]\d{9}$/.test(p)
  const isValidOtp = (o: string) => /^\d{6}$/.test(o)

  const handleSendOtp = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setInfo('')

    if (!isValidPhone(phone)) {
      setError('Enter a valid 10-digit Indian mobile number starting with 6-9')
      return
    }

    setLoading(true)
    try {
      const res = await sendOTP(phone)
      setOtpExpiresInMin(res.expires_in_minutes ?? 5)
      setStep('otp')
      setInfo('Demo mode: the OTP is printed in the API server logs (Render dashboard or local terminal).')
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
      const tokens = await verifyOTP(phone, otp, fullName || undefined)
      setTokens(tokens.access_token, tokens.refresh_token)
      setUser({
        user_id: tokens.user_id,
        phone_number: tokens.phone_number,
        full_name: fullName || undefined,
        kyc_verified: false,
        is_active: true,
        created_at: new Date().toISOString(),
      } as any)
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
            {step === 'phone' ? 'Sign in / Get started' : 'Enter the 6-digit OTP'}
          </h1>
          <p className="text-stone-600">
            {step === 'phone'
              ? 'No password needed — we use mobile OTP, the same way users in tier 2-3 cities prefer.'
              : `Sent to +91 ${phone}. Expires in ${otpExpiresInMin ?? 5} min.`}
          </p>
        </div>

        <Card>
          <Alert
            variant="info"
            message="Demo mode: the OTP is logged in the server console instead of being sent over SMS. Check the Render logs (or your local terminal) to grab it."
          />

          <AnimatePresence mode="wait">
            {step === 'phone' ? (
              <motion.form
                key="phone-step"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
                onSubmit={handleSendOtp}
                className="space-y-4 mt-4"
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
                  label="Mobile number"
                  type="tel"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                  placeholder="10-digit number, starts with 6-9"
                  icon={<Phone className="w-5 h-5" />}
                  inputMode="numeric"
                  autoComplete="tel"
                  required
                />

                <Button type="submit" variant="primary" fullWidth loading={loading}>
                  Send OTP
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
                className="space-y-4 mt-4"
              >
                {error && <Alert variant="error" message={error} />}
                {info && <Alert variant="info" message={info} />}

                <Input
                  label="6-digit OTP"
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
                    setStep('phone')
                    setOtp('')
                    setError('')
                  }}
                  icon={<ArrowLeft className="w-4 h-4" />}
                >
                  Use a different number
                </Button>
              </motion.form>
            )}
          </AnimatePresence>
        </Card>
      </motion.div>
    </div>
  )
}
