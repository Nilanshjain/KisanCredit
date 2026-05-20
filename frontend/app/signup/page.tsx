'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'
import Alert from '@/components/ui/Alert'
import { Sprout, Mail, Lock, User, Phone } from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'

export default function SignupPage() {
  const router = useRouter()
  const { setTokens, setUser } = useAuthStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    full_name: '',
    phone_number: '',
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
    setError('')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log('=== SIGNUP FORM SUBMITTED ===')
    console.log('Form data:', formData)

    setError('')
    setSuccess('')

    // Validate form
    if (!formData.email || !formData.password || !formData.full_name || !formData.phone_number) {
      console.log('Validation failed: Missing fields')
      setError('Please fill in all fields')
      return
    }

    if (formData.password !== formData.confirmPassword) {
      console.log('Validation failed: Passwords do not match')
      setError('Passwords do not match')
      return
    }

    if (formData.password.length < 8) {
      console.log('Validation failed: Password too short')
      setError('Password must be at least 8 characters long')
      return
    }

    // Check password strength
    const hasUpperCase = /[A-Z]/.test(formData.password)
    const hasLowerCase = /[a-z]/.test(formData.password)
    const hasNumber = /\d/.test(formData.password)

    if (!hasUpperCase || !hasLowerCase || !hasNumber) {
      console.log('Validation failed: Password strength requirements not met')
      setError('Password must contain uppercase, lowercase, and a number')
      return
    }

    console.log('All validations passed, making API request...')
    setLoading(true)

    try {
      const payload = {
        email: formData.email,
        password: formData.password,
        full_name: formData.full_name,
        phone_number: formData.phone_number,
      }
      console.log('API Payload:', payload)

      const response = await fetch('http://localhost:8000/api/v1/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      console.log('API Response status:', response.status)
      const data = await response.json()
      console.log('API Response data:', data)

      if (!response.ok) {
        console.error('API Error:', data)
        throw new Error(data.message || data.detail || 'Signup failed')
      }

      // Store tokens in auth store
      console.log('Storing tokens in auth store...')
      setTokens(data.access_token, data.refresh_token)

      // Set user data
      setUser({
        user_id: data.user_id,
        phone_number: data.phone_number,
        full_name: formData.full_name,
        email: formData.email,
      })

      setSuccess('Account created successfully! Redirecting...')
      console.log('Success! Redirecting to dashboard...')

      // Redirect to dashboard after 1 second
      setTimeout(() => {
        router.push('/dashboard')
      }, 1000)
    } catch (err: any) {
      console.error('Signup error:', err)
      const errorMessage = err.message || 'Failed to create account'
      console.error('Setting error message:', errorMessage)
      setError(errorMessage)
    } finally {
      setLoading(false)
      console.log('=== SIGNUP COMPLETE ===')
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
          <h1 className="text-3xl font-bold text-stone-900 mb-2">Create Account</h1>
          <p className="text-stone-600">Join us to access instant credit solutions</p>
        </div>

        <Card>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && <Alert variant="error" message={error} />}
            {success && <Alert variant="success" message={success} />}

            <Input
              label="Full Name"
              type="text"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              placeholder="Enter your full name"
              icon={<User className="w-5 h-5" />}
              required
            />

            <Input
              label="Email"
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="Enter your email"
              icon={<Mail className="w-5 h-5" />}
              required
            />

            <Input
              label="Phone Number"
              type="tel"
              name="phone_number"
              value={formData.phone_number}
              onChange={handleChange}
              placeholder="10-digit mobile number"
              icon={<Phone className="w-5 h-5" />}
              maxLength={10}
              required
            />

            <Input
              label="Password"
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Min 8 characters, 1 uppercase, 1 number"
              icon={<Lock className="w-5 h-5" />}
              required
            />

            <Input
              label="Confirm Password"
              type="password"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              placeholder="Re-enter your password"
              icon={<Lock className="w-5 h-5" />}
              required
            />

            <Button type="submit" variant="primary" fullWidth loading={loading}>
              {loading ? 'Creating Account...' : 'Sign Up'}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-stone-600">
              Already have an account?{' '}
              <Link href="/login" className="text-field-600 hover:text-field-700 font-semibold">
                Login
              </Link>
            </p>
          </div>
        </Card>

        <div className="mt-6 text-center">
          <p className="text-xs text-stone-500">
            By signing up, you agree to our{' '}
            <Link href="/terms" className="underline hover:text-stone-700">
              Terms of Service
            </Link>{' '}
            and{' '}
            <Link href="/privacy" className="underline hover:text-stone-700">
              Privacy Policy
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  )
}
