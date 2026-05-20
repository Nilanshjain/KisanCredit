'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Card, Button, Badge, Alert, Skeleton } from '@/components/ui'
import { ArrowLeft, RefreshCw, Calendar, TrendingUp, TrendingDown, FileText } from 'lucide-react'

interface ApplicationDetail {
  application_id: string
  user_id: string
  status: string
  loan_amount: number
  loan_purpose: string
  submitted_at: string
  processed_at?: string
  processing_time_ms?: number
  extracted_features?: Record<string, number>
  predictions: Array<{
    profitability_score: number
    decision: string
    confidence: number
    prediction_timestamp: string
    model_version?: string
    prediction_latency_ms?: number
  }>
}

export default function ApplicationDetailPage() {
  const params = useParams()
  const router = useRouter()
  const applicationId = params.id as string

  const [application, setApplication] = useState<ApplicationDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchApplicationDetails()
  }, [applicationId])

  const fetchApplicationDetails = async () => {
    try {
      setLoading(true)
      const token = localStorage.getItem('access_token')

      const response = await fetch(
        `http://localhost:8000/api/v1/applications/${applicationId}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (!response.ok) {
        if (response.status === 404) {
          setError('Application not found')
        } else if (response.status === 403) {
          setError('You do not have permission to view this application')
        } else {
          setError('Failed to load application details')
        }
        return
      }

      const data = await response.json()
      setApplication(data)
    } catch (err) {
      setError('Failed to load application details')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 p-8">
        <div className="max-w-6xl mx-auto space-y-6">
          <Skeleton className="h-12 w-64" />
          <Skeleton className="h-96 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    )
  }

  if (error || !application) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 p-8">
        <div className="max-w-6xl mx-auto">
          <Alert variant="error">
            <p>{error || 'Application not found'}</p>
          </Alert>
          <Button
            onClick={() => router.push('/dashboard')}
            className="mt-4"
            leftIcon={<ArrowLeft />}
          >
            Back to Dashboard
          </Button>
        </div>
      </div>
    )
  }

  const latestPrediction = application.predictions[0]
  const creditScore = Math.round(latestPrediction.profitability_score * 100)

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'approved':
      case 'processed':
        return 'success'
      case 'rejected':
        return 'error'
      default:
        return 'warning'
    }
  }

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'approve':
        return 'text-green-600'
      case 'reject':
        return 'text-red-600'
      default:
        return 'text-yellow-600'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <Button
              onClick={() => router.push('/dashboard')}
              variant="ghost"
              leftIcon={<ArrowLeft />}
            >
              Back
            </Button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Application Details
              </h1>
              <p className="text-gray-600 mt-1">
                {application.application_id}
              </p>
            </div>
          </div>

          <Button
            onClick={fetchApplicationDetails}
            variant="outline"
            leftIcon={<RefreshCw />}
          >
            Refresh
          </Button>
        </div>

        {/* Status Card */}
        <Card className="bg-white shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div>
              <p className="text-sm text-gray-600 mb-2">Status</p>
              <Badge variant={getStatusColor(application.status)}>
                {application.status.toUpperCase()}
              </Badge>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-2">Loan Amount</p>
              <p className="text-2xl font-bold text-gray-900">
                ₹{application.loan_amount.toLocaleString('en-IN')}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-2">Credit Score</p>
              <p className="text-2xl font-bold text-gray-900">
                {creditScore}/100
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-2">Decision</p>
              <p className={`text-2xl font-bold ${getDecisionColor(latestPrediction.decision)}`}>
                {latestPrediction.decision.toUpperCase()}
              </p>
            </div>
          </div>
        </Card>

        {/* Timeline */}
        <Card className="bg-white shadow-lg">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Calendar className="w-5 h-5 text-amber-600" />
            Timeline
          </h2>
          <div className="space-y-4">
            <div className="flex gap-4 border-l-4 border-amber-500 pl-4">
              <div className="w-32 text-sm text-gray-600">
                {new Date(application.submitted_at).toLocaleString()}
              </div>
              <div className="flex-1">
                <p className="font-semibold">Application Submitted</p>
                <p className="text-sm text-gray-600">
                  Loan application for ₹{application.loan_amount.toLocaleString('en-IN')}
                </p>
              </div>
            </div>

            {application.processed_at && (
              <div className="flex gap-4 border-l-4 border-green-500 pl-4">
                <div className="w-32 text-sm text-gray-600">
                  {new Date(application.processed_at).toLocaleString()}
                </div>
                <div className="flex-1">
                  <p className="font-semibold">Application Processed</p>
                  <p className="text-sm text-gray-600">
                    Processing time: {application.processing_time_ms?.toFixed(2)}ms
                    {latestPrediction.prediction_latency_ms &&
                      ` (ML prediction: ${latestPrediction.prediction_latency_ms.toFixed(2)}ms)`
                    }
                  </p>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* ML Prediction Details */}
        <Card className="bg-white shadow-lg">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-amber-600" />
            ML Prediction Analysis
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Profitability Score</p>
              <p className="text-3xl font-bold text-amber-700">
                {(latestPrediction.profitability_score * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Model Confidence</p>
              <p className="text-3xl font-bold text-blue-700">
                {(latestPrediction.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Model Version</p>
              <p className="text-3xl font-bold text-green-700">
                {latestPrediction.model_version || '1.0'}
              </p>
            </div>
          </div>
        </Card>

        {/* Feature Breakdown */}
        {application.extracted_features && (
          <Card className="bg-white shadow-lg">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5 text-amber-600" />
              Extracted Features (Top 12)
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(application.extracted_features)
                .slice(0, 12)
                .map(([feature, value]) => (
                  <div key={feature} className="border-l-4 border-amber-500 pl-4 py-2">
                    <p className="text-xs text-gray-600 truncate uppercase" title={feature}>
                      {feature.replace(/_/g, ' ')}
                    </p>
                    <p className="text-lg font-semibold text-gray-900">
                      {typeof value === 'number' ? value.toFixed(3) : value}
                    </p>
                  </div>
                ))}
            </div>
            <p className="text-sm text-gray-500 mt-4">
              {Object.keys(application.extracted_features).length} total features extracted
            </p>
          </Card>
        )}

        {/* Prediction History */}
        {application.predictions.length > 1 && (
          <Card className="bg-white shadow-lg">
            <h2 className="text-xl font-bold mb-4">Prediction History</h2>
            <div className="space-y-3">
              {application.predictions.map((pred, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex-1">
                    <p className="font-medium">
                      {new Date(pred.prediction_timestamp).toLocaleString()}
                    </p>
                    <p className="text-sm text-gray-600">
                      Model {pred.model_version || '1.0'}
                    </p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Score</p>
                      <p className="font-bold">{(pred.profitability_score * 100).toFixed(1)}%</p>
                    </div>
                    <Badge variant={
                      pred.decision === 'approve' ? 'success' :
                      pred.decision === 'reject' ? 'error' : 'warning'
                    }>
                      {pred.decision.toUpperCase()}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}
