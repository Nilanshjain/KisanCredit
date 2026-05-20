import { Button } from '@/components/ui'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-8 py-16">
        <Link href="/">
          <Button variant="ghost" icon={<ArrowLeft className="w-4 h-4" />} className="mb-8">
            Back to Home
          </Button>
        </Link>

        <h1 className="text-4xl font-bold mb-8">Terms of Service</h1>

        <div className="prose prose-lg max-w-none">
          <p className="text-gray-600 mb-8">
            Last updated: {new Date().toLocaleDateString()}
          </p>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">1. Acceptance of Terms</h2>
            <p className="text-gray-700 leading-relaxed">
              By accessing and using KisanCredit, you accept and agree to be bound by the terms and provisions of this agreement.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">2. Use of Service</h2>
            <p className="text-gray-700 leading-relaxed">
              KisanCredit provides an alternative credit scoring platform using machine learning. You agree to:
            </p>
            <ul className="list-disc pl-6 text-gray-700 space-y-2">
              <li>Provide accurate and truthful information</li>
              <li>Use the service only for lawful purposes</li>
              <li>Not attempt to manipulate or circumvent the ML scoring system</li>
              <li>Maintain the security of your account credentials</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">3. Privacy & Data</h2>
            <p className="text-gray-700 leading-relaxed">
              We collect and use your data as described in our Privacy Policy. By using KisanCredit, you consent to our data collection and usage practices.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">4. Limitation of Liability</h2>
            <p className="text-gray-700 leading-relaxed">
              KisanCredit provides credit scoring as-is without warranties. We are not liable for decisions made based on our scoring system.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-bold mb-4">5. Contact</h2>
            <p className="text-gray-700">
              Questions about these terms? Contact:{' '}
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
