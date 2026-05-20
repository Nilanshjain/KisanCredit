'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion, useScroll, useTransform } from 'framer-motion'
import {
  Sprout,
  ArrowRight,
  CheckCircle2,
  Zap,
  Shield,
  TrendingUp,
  Users,
  Smartphone,
  Clock,
  Sparkles,
  BarChart3,
  Lock,
  Globe
} from 'lucide-react'
import { useAuthStore } from '@/lib/authStore'
import { Button, Card } from '@/components/ui'

// Animated counter component
function AnimatedCounter({ end, duration = 2000, suffix = '' }: { end: number; duration?: number; suffix?: string }) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let startTime: number
    let animationFrame: number

    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime
      const progress = Math.min((currentTime - startTime) / duration, 1)

      setCount(Math.floor(progress * end))

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate)
      }
    }

    animationFrame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationFrame)
  }, [end, duration])

  return <span>{count}{suffix}</span>
}

export default function HomePage() {
  const { isAuthenticated } = useAuthStore()
  const [isScrolled, setIsScrolled] = useState(false)
  const { scrollYProgress } = useScroll()

  // Parallax effect for hero
  const heroY = useTransform(scrollYProgress, [0, 0.5], [0, 100])
  const heroOpacity = useTransform(scrollYProgress, [0, 0.3], [1, 0])

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const features = [
    {
      icon: Zap,
      title: 'Fast ML Predictions',
      description: 'AI model analyzes your financial data and provides instant credit scoring',
      color: 'harvest',
    },
    {
      icon: Shield,
      title: 'Secure Platform',
      description: 'Built with FastAPI and PostgreSQL with authentication and data validation',
      color: 'field',
    },
    {
      icon: BarChart3,
      title: 'Alternative Credit Scoring',
      description: 'AI-powered creditworthiness assessment using alternative data sources',
      color: 'harvest',
    },
    {
      icon: Smartphone,
      title: 'Mobile-First Design',
      description: 'Optimized for smartphones with simple, accessible interface',
      color: 'field',
    },
  ]

  const stats = [
    { value: 47, suffix: '', label: 'Features Analyzed', subtext: 'Income, expense, social metrics' },
    { value: 60, suffix: 's', label: 'Instant Decisions', subtext: 'Fast ML-powered approval' },
    { value: 50, suffix: 'ms', label: 'P95 Latency', subtext: 'Production-grade performance' },
    { value: 24, suffix: '/7', label: 'Always Available', subtext: 'Cloud-hosted platform' },
  ]

  const steps = [
    {
      number: '01',
      title: 'Sign Up / Login',
      description: 'Create an account using phone number verification.',
      icon: Smartphone,
    },
    {
      number: '02',
      title: 'Submit Application',
      description: 'Fill out the loan application form with financial and personal details.',
      icon: BarChart3,
    },
    {
      number: '03',
      title: 'ML Prediction',
      description: 'System generates 47 features and runs ML model to calculate credit score.',
      icon: Zap,
    },
  ]

  return (
    <div className="min-h-screen bg-rural-pattern relative">
      {/* Hero Section */}
      <motion.section
        style={{ y: heroY, opacity: heroOpacity }}
        className="relative pt-32 pb-20 px-4 overflow-hidden bg-gradient-to-br from-harvest-50 via-amber-50 to-field-50"
      >
        {/* Decorative elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Wheat/Grain stalks decoration */}
          <div className="absolute top-0 left-0 w-full h-full opacity-5">
            <svg className="absolute top-10 left-10 w-32 h-32 text-harvest-600" viewBox="0 0 100 100" fill="currentColor">
              <path d="M50 10 L50 90 M45 20 Q40 25 45 30 M55 20 Q60 25 55 30 M45 35 Q40 40 45 45 M55 35 Q60 40 55 45 M45 50 Q40 55 45 60 M55 50 Q60 55 55 60 M45 65 Q40 70 45 75 M55 65 Q60 70 55 75" stroke="currentColor" strokeWidth="2" fill="none"/>
            </svg>
            <svg className="absolute top-20 right-20 w-40 h-40 text-field-600" viewBox="0 0 100 100" fill="currentColor" opacity="0.8">
              <path d="M50 10 L50 90 M45 20 Q40 25 45 30 M55 20 Q60 25 55 30 M45 35 Q40 40 45 45 M55 35 Q60 40 55 45 M45 50 Q40 55 45 60 M55 50 Q60 55 55 60 M45 65 Q40 70 45 75 M55 65 Q60 70 55 75" stroke="currentColor" strokeWidth="2" fill="none"/>
            </svg>
            <svg className="absolute bottom-20 left-1/4 w-36 h-36 text-harvest-500" viewBox="0 0 100 100" fill="currentColor" opacity="0.6">
              <path d="M50 10 L50 90 M45 20 Q40 25 45 30 M55 20 Q60 25 55 30 M45 35 Q40 40 45 45 M55 35 Q60 40 55 45 M45 50 Q40 55 45 60 M55 50 Q60 55 55 60 M45 65 Q40 70 45 75 M55 65 Q60 70 55 75" stroke="currentColor" strokeWidth="2" fill="none"/>
            </svg>
          </div>
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              rotate: [0, 90, 0],
            }}
            transition={{ duration: 20, repeat: Infinity }}
            className="absolute top-20 right-20 w-96 h-96 bg-harvest-200 rounded-full opacity-20 blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1, 1.3, 1],
              rotate: [0, -90, 0],
            }}
            transition={{ duration: 25, repeat: Infinity }}
            className="absolute bottom-20 left-20 w-96 h-96 bg-field-200 rounded-full opacity-20 blur-3xl"
          />
        </div>

        <div className="container-custom relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/60 backdrop-blur-sm rounded-full border border-harvest-200 mb-8"
            >
              <Sparkles className="w-4 h-4 text-harvest-600" />
              <span className="text-sm font-medium text-stone-700">AI-Powered Credit Scoring</span>
            </motion.div>

            {/* Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-5xl md:text-7xl font-bold text-stone-900 mb-6 leading-tight"
            >
              AI-Powered{' '}
              <span className="text-gradient-harvest">Credit Scoring</span>
              <br />
              for Rural India
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-xl md:text-2xl text-stone-600 mb-12 max-w-3xl mx-auto leading-relaxed"
            >
              Alternative credit scoring platform for underserved communities using AI and machine learning. Get loan decisions in 60 seconds using SMS, UPI, and contact data—no traditional credit history required.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            >
              <Button
                variant="primary"
                size="lg"
                onClick={() => window.location.href = '/apply'}
                icon={<ArrowRight className="w-5 h-5" />}
                iconPosition="right"
              >
                Apply for Loan Now
              </Button>
              <Button
                variant="secondary"
                size="lg"
                onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
              >
                See How It Works
              </Button>
            </motion.div>

            {/* Trust badges */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="mt-16 flex flex-wrap justify-center gap-8 text-stone-600"
            >
              {[
                { icon: Shield, text: 'Bank-Grade Security' },
                { icon: Lock, text: 'Data Protected' },
                { icon: Zap, text: 'Instant Approval' },
              ].map((item, index) => (
                <div key={index} className="flex items-center gap-2">
                  <item.icon className="w-5 h-5 text-harvest-600" />
                  <span className="text-sm font-medium">{item.text}</span>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Stats Section */}
      <section className="py-16 px-4 bg-white/80 backdrop-blur-sm relative">
        <div className="absolute inset-0 bg-gradient-to-b from-harvest-50/30 to-transparent pointer-events-none"></div>
        <div className="container-custom">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card padding="md" className="text-center hover:shadow-soft-lg transition-shadow">
                  <div className="text-4xl md:text-5xl font-bold text-harvest-600 mb-2">
                    <AnimatedCounter end={stat.value} suffix={stat.suffix} />
                  </div>
                  <div className="text-sm font-semibold text-stone-900 mb-1">{stat.label}</div>
                  <div className="text-xs text-stone-500">{stat.subtext}</div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="section bg-field-pattern relative">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-stone-900 mb-4">
              Why KisanCredit?
            </h2>
            <p className="text-xl text-stone-600 max-w-2xl mx-auto">
              Built for communities traditional banks often overlook
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card variant="hover" padding="lg" className="h-full">
                  <div className={
                    feature.color === 'harvest'
                      ? 'inline-flex p-3 rounded-xl bg-harvest-50 text-harvest-600 mb-4'
                      : 'inline-flex p-3 rounded-xl bg-field-50 text-field-600 mb-4'
                  }>
                    <feature.icon className="w-6 h-6" />
                  </div>
                  <h3 className="text-xl font-bold text-stone-900 mb-2">{feature.title}</h3>
                  <p className="text-stone-600">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="section bg-white/90 backdrop-blur-sm relative">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-harvest-400 to-transparent"></div>
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-stone-900 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-stone-600 max-w-2xl mx-auto">
              Get your loan in three simple steps
            </p>
          </motion.div>

          <div className="max-w-4xl mx-auto">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                className="relative"
              >
                <div className="flex gap-6 mb-12">
                  {/* Number circle */}
                  <div className="flex-shrink-0">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-harvest text-stone-900 flex items-center justify-center font-bold text-xl shadow-soft-lg">
                      {step.number}
                    </div>
                  </div>

                  {/* Content */}
                  <Card padding="lg" className="flex-1">
                    <div className="flex items-start gap-4">
                      <div className="p-3 bg-harvest-50 rounded-xl flex-shrink-0">
                        <step.icon className="w-6 h-6 text-harvest-600" />
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold text-stone-900 mb-2">{step.title}</h3>
                        <p className="text-stone-600 leading-relaxed">{step.description}</p>
                      </div>
                    </div>
                  </Card>
                </div>

                {/* Connecting line */}
                {index < steps.length - 1 && (
                  <div className="absolute left-8 top-20 bottom-0 w-0.5 bg-stone-200 -z-10" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Trust Section */}
      <section className="section bg-gradient-to-br from-harvest-400 via-harvest-500 to-harvest-600">
        <div className="container-custom">
          <div className="max-w-3xl mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
            >
              <div className="inline-flex p-4 bg-white/30 backdrop-blur-sm rounded-2xl mb-6">
                <CheckCircle2 className="w-12 h-12 text-stone-900" />
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 text-stone-900">
                Try the Demo
              </h2>
              <p className="text-xl mb-8 text-stone-800">
                Experience how machine learning can be used for alternative credit scoring. This is a prototype for educational purposes.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={() => window.location.href = '/login'}
                  icon={<ArrowRight className="w-5 h-5" />}
                  iconPosition="right"
                >
                  Try Demo
                </Button>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-stone-100 text-stone-700 py-12 px-4 border-t border-stone-200">
        <div className="container-custom">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-2">
              <Sprout className="w-6 h-6 text-harvest-500" />
              <span className="text-xl font-bold text-stone-900">KisanCredit</span>
            </div>

            <div className="flex flex-wrap justify-center gap-6 text-sm">
              <Link href="/terms" className="hover:text-stone-900 transition-colors font-medium">Terms of Service</Link>
              <Link href="/privacy" className="hover:text-stone-900 transition-colors font-medium">Privacy Policy</Link>
              <Link href="/contact" className="hover:text-stone-900 transition-colors font-medium">Contact Us</Link>
            </div>

            <div className="text-sm text-stone-600">
              © 2025 KisanCredit. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
