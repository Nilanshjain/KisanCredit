'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Card from '@/components/ui/Card'
import Alert from '@/components/ui/Alert'
import { Sprout, Mail, Lock } from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'

export default function LoginPage() {
  const router = useRouter()
  const { setTokens, setUser } = useAuthStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const [formData, setFormData] = useState({
    email: '',
    password: '',
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
    console.log('=== LOGIN FORM SUBMITTED ===')
    console.log('Form data:', formData)

    setError('')
    setSuccess('')

    // Validate form
    if (!formData.email || !formData.password) {
      console.log('Validation failed: Missing fields')
      setError('Please fill in all fields')
      return
    }

    console.log('Validation passed, making API request...')
    setLoading(true)

    try {
      const payload = {
        email: formData.email,
        password: formData.password,
      }
      console.log('API Payload:', payload)

      const response = await fetch('http://localhost:8000/api/v1/auth/login', {
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
        throw new Error(data.message || data.detail || 'Login failed')
      }

      // Store tokens in auth store
      console.log('Storing tokens in auth store...')
      setTokens(data.access_token, data.refresh_token)

      // Set user data if available
      if (data.user_id) {
        setUser({
          user_id: data.user_id,
          phone_number: data.phone_number,
          full_name: null,
          email: formData.email,
        })
      }

      setSuccess('Login successful! Redirecting...')
      console.log('Success! Redirecting to dashboard...')

      // Redirect to dashboard after 1 second
      setTimeout(() => {
        router.push('/dashboard')
      }, 1000)
    } catch (err: any) {
      console.error('Login error:', err)
      const errorMessage = err.message || 'Failed to login'
      console.error('Setting error message:', errorMessage)
      setError(errorMessage)
    } finally {
      setLoading(false)
      console.log('=== LOGIN COMPLETE ===')
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
          <h1 className="text-3xl font-bold text-stone-900 mb-2">Welcome Back</h1>
          <p className="text-stone-600">Login to access your account</p>
        </div>

        <Card>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && <Alert variant="error" message={error} />}
            {success && <Alert variant="success" message={success} />}

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
              label="Password"
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Enter your password"
              icon={<Lock className="w-5 h-5" />}
              required
            />

            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-field-600 border-stone-300 rounded focus:ring-field-500"
                />
                <span className="ml-2 text-sm text-stone-600">Remember me</span>
              </label>
              <Link href="/forgot-password" className="text-sm text-field-600 hover:text-field-700 font-semibold">
                Forgot password?
              </Link>
            </div>

            <Button type="submit" variant="primary" fullWidth loading={loading}>
              {loading ? 'Logging in...' : 'Login'}
            </Button>
          </form>

          <div className="mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-stone-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white text-stone-500">Or continue with</span>
              </div>
            </div>

            <div className="mt-6">
              <Link href="/send-otp">
                <Button variant="outline" fullWidth>
                  Login with OTP
                </Button>
              </Link>
            </div>
          </div>

          <div className="mt-6 text-center">
            <p className="text-sm text-stone-600">
              Don't have an account?{' '}
              <Link href="/signup" className="text-field-600 hover:text-field-700 font-semibold">
                Sign Up
              </Link>
            </p>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}
