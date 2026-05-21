/**
 * Applications + predictions API client.
 *
 * Feature engineering happens server-side via /applications/simple — the
 * frontend only collects the simplified form fields. The deprecated
 * client-side `generateFeaturesFromApplication` was removed in Phase 1.
 */

import { useAuthStore } from './authStore';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface LoanApplicationData {
  fullName: string;
  mobile: string;
  dob: string;
  gender: string;
  occupation: string;
  pincode: string;
  loanAmount: number;
  loanPurpose: string;
  monthlyIncome: number;
  monthlyExpenses: number;
  // Form also collects these but the server synthesises richer signals from
  // monthly_income/expenses, so they're not sent. Kept on the type for the
  // existing form UI.
  upiTransactions?: number;
  totalContacts?: number;
  smsCount?: number;
  appUsageMinutes?: number;
}

export interface PredictionResponse {
  application_id: string;
  profitability_score: number;
  credit_score: number;       // profitability_score * 100, for UI
  confidence: number;
  decision: 'approve' | 'reject' | 'manual_review';
  processing_time_ms: number;
  timestamp: string;
  top_features: Array<{ name: string; value: number }>;
}

/** Async-lifecycle submit response — no decision yet, poll the timeline. */
export interface ApplicationSubmittedResponse {
  application_id: string;
  status: 'submitted';
  submitted_at: string;
  message: string;
}

export interface ApplicationTimelineEvent {
  from_status: string | null;
  to_status: string;
  actor_type: 'system' | 'user' | 'admin';
  actor_id: string | null;
  reason: string | null;
  occurred_at: string;
}

export interface ApplicationTimelineResponse {
  application_id: string;
  current_status: string;
  events: ApplicationTimelineEvent[];
}

/** Status values that mean "still processing — keep polling". */
export const NON_TERMINAL_STATUSES = new Set(['submitted', 'under_review']);

/** Read the current access token from the persisted Zustand store. */
export function getAccessToken(): string | null {
  return useAuthStore.getState().accessToken;
}

function authHeader(): Record<string, string> {
  const token = getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Submit a loan application via the simplified form endpoint.
 *
 * Server enqueues an async lifecycle: submitted -> under_review (5s) ->
 * decided (10s more, with inference). The response comes back immediately
 * with application_id and status='submitted'. Caller should redirect to the
 * detail page and poll {@link fetchTimeline} for live updates.
 */
export async function submitApplication(data: LoanApplicationData): Promise<ApplicationSubmittedResponse> {
  const body = {
    name: data.fullName,
    mobile: data.mobile,
    date_of_birth: data.dob,
    gender: data.gender.toLowerCase(),
    pincode: data.pincode,
    occupation: data.occupation,
    loan_amount: data.loanAmount,
    loan_purpose: data.loanPurpose,
    monthly_income: data.monthlyIncome,
    monthly_expenses: data.monthlyExpenses,
  };

  const response = await fetch(`${API_BASE_URL}/applications/simple`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`;
    try {
      const err = await response.json();
      detail = err.detail || err.message || detail;
    } catch {
      // body not JSON — keep the status-based detail
    }
    throw new Error(detail);
  }

  return response.json();
}

// ─────────────────────── Admin (lender/operator) ───────────────────────

export interface AdminMetricsOverview {
  window_hours: number;
  total_predictions: number;
  avg_score: number;
  avg_confidence: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  decisions: Array<{ decision: string; count: number }>;
  score_histogram: Array<{ range_low: number; range_high: number; count: number }>;
  pending_review_count: number;
  recent_buffer_size: number;
  drift_baseline_available: boolean;
}

export interface AdminApplicationSummary {
  application_id: string;
  user_id: string;
  user_phone: string | null;
  loan_amount: number;
  loan_purpose: string;
  status: string;
  submitted_at: string;
  decision: string | null;
  score: number | null;
}

export interface AdminApplicationsList {
  total: number;
  limit: number;
  offset: number;
  applications: AdminApplicationSummary[];
}

export interface AdminApplicationDetail {
  application_id: string;
  user_id: string;
  user_phone: string | null;
  user_full_name: string | null;
  loan_amount: number;
  loan_purpose: string;
  status: string;
  submitted_at: string;
  processed_at: string | null;
  processing_time_ms: number | null;
  extracted_features: Record<string, number> | null;
  latest_prediction: {
    profitability_score: number;
    confidence: number;
    decision: string;
    model_version: string | null;
    prediction_latency_ms: number | null;
    prediction_timestamp: string;
  } | null;
  timeline: ApplicationTimelineEvent[];
}

export interface AdminDriftFeature {
  feature: string;
  psi: number | null;
  severity: 'stable' | 'moderate' | 'significant' | 'unknown' | 'no_data';
  n_current: number;
  baseline_available: boolean;
}

export interface AdminDriftResponse {
  n_recent_inputs: number;
  baseline_available: boolean;
  summary: Record<string, number>;
  features: AdminDriftFeature[];
}

async function adminGet<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE_URL}/admin${path}`, {
    headers: { ...authHeader() }, cache: 'no-store',
  });
  if (!r.ok) {
    if (r.status === 403) throw new Error('Admin role required.');
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `request failed (${r.status})`);
  }
  return r.json();
}

export const fetchAdminMetrics = () => adminGet<AdminMetricsOverview>('/metrics/overview');
export const fetchAdminApplications = (statusFilter?: string, limit = 50, offset = 0) => {
  const params = new URLSearchParams();
  if (statusFilter) params.set('status_filter', statusFilter);
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  return adminGet<AdminApplicationsList>(`/applications?${params}`);
};
export const fetchAdminApplicationDetail = (applicationId: string) =>
  adminGet<AdminApplicationDetail>(`/applications/${encodeURIComponent(applicationId)}`);
export const fetchAdminDrift = () => adminGet<AdminDriftResponse>('/drift');

export async function adminOverrideDecision(
  applicationId: string, decision: 'approve' | 'reject' | 'manual_review', reason: string,
): Promise<{ application_id: string; new_status: string; overridden_decision: string; occurred_at: string }> {
  const r = await fetch(
    `${API_BASE_URL}/admin/applications/${encodeURIComponent(applicationId)}/override`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeader() },
      body: JSON.stringify({ decision, reason }),
    },
  );
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Override failed (${r.status})`);
  }
  return r.json();
}

// ────────────────────────────────────────────────────────────────────────

export async function fetchTimeline(applicationId: string): Promise<ApplicationTimelineResponse> {
  const response = await fetch(
    `${API_BASE_URL}/applications/${encodeURIComponent(applicationId)}/timeline`,
    { headers: { ...authHeader() }, cache: 'no-store' },
  );
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Timeline fetch failed: ${response.status}`);
  }
  return response.json();
}

export function getDecision(creditScore: number): 'APPROVED' | 'MANUAL_REVIEW' | 'REJECTED' {
  if (creditScore >= 60) return 'APPROVED';
  if (creditScore >= 40) return 'MANUAL_REVIEW';
  return 'REJECTED';
}

export function calculateEMI(principal: number, annualRate: number, tenureMonths: number): number {
  const monthlyRate = annualRate / 12 / 100;
  const emi =
    (principal * monthlyRate * Math.pow(1 + monthlyRate, tenureMonths)) /
    (Math.pow(1 + monthlyRate, tenureMonths) - 1);
  return Math.ceil(emi);
}

/** Fetch the SHAP-based explanation for an existing application. */
export interface ExplanationContributor {
  feature: string;
  value: number;
  contribution: number;
  importance: number;
}

export interface NaturalLanguageExplanation {
  text: string;
  suggestion: string;
  language: 'en' | 'hi';
  source: 'gemini' | 'template';
  cached: boolean;
}

export interface ApplicationExplanation {
  application_id: string;
  profitability_score: number;
  decision: 'approve' | 'reject' | 'manual_review';
  base_value: number;
  top_contributors: ExplanationContributor[];
  explanation_timestamp: string;
  natural_language: NaturalLanguageExplanation | null;
}

export interface CounterfactualChange {
  feature: string;
  display_label: string;
  display_unit: string;
  current: number;
  suggested: number;
  delta_score: number;
  new_score: number;
}

export interface CounterfactualResult {
  application_id: string;
  starting_score: number;
  final_score: number;
  reachable: boolean;
  changes: CounterfactualChange[];
  natural_language: NaturalLanguageExplanation | null;
}

export async function fetchExplanation(applicationId: string, language: 'en' | 'hi' = 'en'): Promise<ApplicationExplanation> {
  const r = await fetch(
    `${API_BASE_URL}/predictions/${encodeURIComponent(applicationId)}/explain?language=${language}`,
    { headers: { ...authHeader() }, cache: 'no-store' },
  );
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Explanation failed: ${r.status}`);
  }
  return r.json();
}

export async function fetchCounterfactual(applicationId: string, language: 'en' | 'hi' = 'en'): Promise<CounterfactualResult> {
  const r = await fetch(
    `${API_BASE_URL}/predictions/${encodeURIComponent(applicationId)}/counterfactual?language=${language}`,
    { headers: { ...authHeader() }, cache: 'no-store' },
  );
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Counterfactual failed: ${r.status}`);
  }
  return r.json();
}
