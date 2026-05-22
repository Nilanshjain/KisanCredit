/**
 * Authentication API client for KisanCredit
 */

import { authedFetch } from './authedFetch';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface SendOTPResponse {
  success: boolean;
  message: string;
  email: string;
  expires_in_minutes: number;
  /** Present only when the backend runs in DEMO_MODE — the OTP echoed back
   *  so the public demo is usable without inbox access. */
  demo_otp?: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user_id: string;
  email: string;
  is_new_user: boolean;
}

export interface UserProfile {
  user_id: string;
  email?: string;
  phone_number?: string;   // optional — email-OTP users have no phone
  full_name?: string;
  date_of_birth?: string;
  address?: string;
  city?: string;
  state?: string;
  pincode?: string;
  employment_type?: string;
  monthly_income?: number;
  kyc_verified: boolean;
  is_active: boolean;
  /** 'user' (applicant) or 'admin' (lender/operator). Gates /admin/* routes. */
  role?: 'user' | 'admin';
  created_at: string;
}

export interface ApplicationSummary {
  id: string;
  application_id: string;
  loan_amount: number;
  loan_purpose: string;
  status: string;
  submitted_at: string;
  processed_at?: string;
}

export interface UserApplicationsResponse {
  total: number;
  applications: ApplicationSummary[];
}

/**
 * Send a one-time code to an email address.
 */
export async function sendOTP(email: string): Promise<SendOTPResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/send-otp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Failed to send OTP');
  }
  return response.json();
}

/**
 * Verify the emailed OTP and login/signup.
 */
export async function verifyOTP(
  email: string,
  otp: string,
  fullName?: string,
): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/verify-otp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, otp, full_name: fullName }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Failed to verify OTP');
  }
  return response.json();
}

/**
 * Refresh access token
 */
export async function refreshToken(refreshToken: string): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });

  if (!response.ok) {
    throw new Error('Failed to refresh token');
  }

  return response.json();
}

/**
 * Validate current token
 */
export async function validateToken(accessToken: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/auth/validate`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });

  if (!response.ok) {
    throw new Error('Token validation failed');
  }

  return response.json();
}

/**
 * Logout user
 */
export async function logout(accessToken: string): Promise<void> {
  await fetch(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });
}

/**
 * Get current user profile. Token handling (including refresh-on-401) lives
 * in authedFetch.
 */
export async function getCurrentUser(): Promise<UserProfile> {
  const response = await authedFetch(`${API_BASE_URL}/users/me`, { method: 'GET' });
  if (!response.ok) {
    throw new Error('Failed to fetch user profile');
  }
  return response.json();
}

/**
 * Update user profile
 */
export async function updateUserProfile(data: Partial<UserProfile>): Promise<UserProfile> {
  const response = await authedFetch(`${API_BASE_URL}/users/me`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error('Failed to update profile');
  }
  return response.json();
}

/**
 * Get the signed-in user's loan applications.
 */
export async function getUserApplications(
  statusFilter?: string
): Promise<UserApplicationsResponse> {
  const url = new URL(`${API_BASE_URL}/users/me/applications`);
  if (statusFilter) {
    url.searchParams.set('status_filter', statusFilter);
  }
  const response = await authedFetch(url.toString(), { method: 'GET' });
  if (!response.ok) {
    throw new Error('Failed to fetch applications');
  }
  return response.json();
}
