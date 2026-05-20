import { Button } from '@/components/ui'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-8 py-16">
        <Link href="/">
          <Button variant="ghost" icon={<ArrowLeft className="w-4 h-4" />} className="mb-8">
            Back to Home
          </Button>
        </Link>

        <h1 className="text-4xl font-bold mb-8">Privacy Policy</h1>

        <div className="prose prose-lg max-w-none">
          <p className="text-gray-600 mb-8">
            Last updated: {new Date().toLocaleDateString()}
          </p>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">1. Information We Collect</h2>
            <p className="text-gray-700 leading-relaxed">
              We collect information you provide when using KisanCredit:
            </p>
            <ul className="list-disc pl-6 text-gray-700 space-y-2">
              <li><strong>Personal Information:</strong> Name, phone number, date of birth</li>
              <li><strong>Financial Data:</strong> SMS transactions, UPI history, monthly income/expenses</li>
              <li><strong>Contact Data:</strong> Contact list metadata for social network analysis</li>
              <li><strong>Location Data:</strong> Location patterns for stability assessment</li>
              <li><strong>Behavioral Data:</strong> App usage patterns and financial behavior</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">2. How We Use Your Data</h2>
            <p className="text-gray-700 leading-relaxed">
              Your data is used to:
            </p>
            <ul className="list-disc pl-6 text-gray-700 space-y-2">
              <li>Generate credit scores using our ML model</li>
              <li>Assess loan profitability and creditworthiness</li>
              <li>Improve our machine learning algorithms</li>
              <li>Provide personalized loan recommendations</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">3. Data Security</h2>
            <p className="text-gray-700 leading-relaxed">
              We implement industry-standard security measures:
            </p>
            <ul className="list-disc pl-6 text-gray-700 space-y-2">
              <li>Encrypted data transmission (HTTPS/TLS)</li>
              <li>Secure cloud storage with encryption at rest</li>
              <li>JWT-based authentication</li>
              <li>Regular security audits</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">4. Data Sharing</h2>
            <p className="text-gray-700 leading-relaxed">
              We <strong>do NOT</strong> sell your personal data. We may share data with:
            </p>
            <ul className="list-disc pl-6 text-gray-700 space-y-2">
              <li>Financial partners for loan processing (with your consent)</li>
              <li>Service providers who help operate our platform</li>
              <li>Law enforcement when required by law</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">5. Your Rights</h2>
            <p className="text-gray-700 leading-relaxed">
              You have the right to:
            </p>
            <ul className="list-disc pl-6 text-gray-700 space-y-2">
              <li>Access your data</li>
              <li>Request data correction</li>
              <li>Request data deletion</li>
              <li>Opt out of data collection</li>
              <li>Download your data</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">6. Contact Us</h2>
            <p className="text-gray-700">
              Privacy questions or data requests:{' '}
              <a href="mailto:nilanshjain0306@gmail.com" className="text-amber-600 hover:underline">
                nilanshjain0306@gmail.com
              </a>
            </p>
          </section>
        </div>
      </div>
    </div>
  )
}
