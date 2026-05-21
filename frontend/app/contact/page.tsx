import { Mail, Github, Linkedin } from 'lucide-react'
import { Card, Button } from '@/components/ui'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function ContactPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50">
      <div className="max-w-4xl mx-auto px-8 py-16">
        <Link href="/">
          <Button variant="ghost" icon={<ArrowLeft className="w-4 h-4" />} className="mb-8">
            Back to Home
          </Button>
        </Link>

        <h1 className="text-4xl font-bold mb-4 text-center">Contact Us</h1>
        <p className="text-center text-gray-600 mb-12">
          Get in touch about KisanCredit
        </p>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-white shadow-lg">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-amber-100 rounded-lg">
                <Mail className="w-6 h-6 text-amber-700" />
              </div>
              <div>
                <h3 className="font-bold text-lg mb-2">Email</h3>
                <p className="text-gray-600 mb-3">
                  For questions about the platform
                </p>
                <a
                  href="mailto:nilanshjain0306@gmail.com"
                  className="text-amber-600 hover:underline"
                >
                  nilanshjain0306@gmail.com
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-white shadow-lg">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-amber-100 rounded-lg">
                <Github className="w-6 h-6 text-amber-700" />
              </div>
              <div>
                <h3 className="font-bold text-lg mb-2">GitHub</h3>
                <p className="text-gray-600 mb-3">
                  View source code and documentation
                </p>
                <a
                  href="https://github.com/yourusername/KisanCredit"
                  className="text-amber-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  github.com/kisancredit
                </a>
              </div>
            </div>
          </Card>
        </div>

        <div className="mt-12 text-center">
          <Card className="bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg">
            <h3 className="text-xl font-bold mb-2">About KisanCredit</h3>
            <p className="mb-4">
              AI-powered alternative credit scoring platform for rural India. Built with FastAPI, LightGBM, Next.js, and deployed on cloud infrastructure.
            </p>
            <Link href="/apply">
              <Button className="bg-white text-amber-600 hover:bg-amber-50">
                Apply for Loan
              </Button>
            </Link>
          </Card>
        </div>
      </div>
    </div>
  )
}
