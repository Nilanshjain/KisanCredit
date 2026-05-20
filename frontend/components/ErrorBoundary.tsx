'use client'

import React from 'react'
import { Alert, Button } from './ui'
import { RefreshCw, Home } from 'lucide-react'

interface Props {
  children: React.ReactNode
  fallback?: React.ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 flex items-center justify-center p-8">
          <div className="max-w-md w-full">
            <Alert variant="error">
              <div className="space-y-4">
                <div>
                  <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
                  <p className="text-sm text-gray-600">
                    We're sorry, an unexpected error occurred. Please try refreshing the page.
                  </p>
                </div>

                {process.env.NODE_ENV === 'development' && this.state.error && (
                  <div className="bg-red-50 border border-red-200 rounded p-3">
                    <p className="text-xs font-mono text-red-800">
                      {this.state.error.message}
                    </p>
                  </div>
                )}

                <div className="flex gap-2">
                  <Button
                    onClick={() => window.location.reload()}
                    leftIcon={<RefreshCw />}
                    className="flex-1"
                  >
                    Refresh Page
                  </Button>
                  <Button
                    onClick={() => window.location.href = '/'}
                    leftIcon={<Home />}
                    variant="outline"
                    className="flex-1"
                  >
                    Go Home
                  </Button>
                </div>
              </div>
            </Alert>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
