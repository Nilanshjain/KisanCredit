/**
 * Authentication API client for KisanCredit
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface SendOTPResponse {
  success: boolean;
  message: string;
  phone_number: string;
  expires_in_minutes: number;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user_id: string;
  phone_number: string;
  is_new_user: boolean;
}

export interface UserProfile {
  user_id: string;
  phone_number: string;
  email?: string;
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
 * Send OTP to phone number
 */
export async function sendOTP(phoneNumber: string): Promise<SendOTPResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/send-otp`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ phone_number: phoneNumber }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to send OTP');
  }

  return response.json();
}

/**
 * Verify OTP and login/signup
 */
export async function verifyOTP(
  phoneNumber: string,
  otp: string,
  fullName?: string
): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/verify-otp`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      phone_number: phoneNumber,
      otp,
      full_name: fullName,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
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
 * Get current user profile
 */
export async function getCurrentUser(accessToken: string): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/users/me`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });

  if (!response.ok) {
    throw new Error('Failed to fetch user profile');
  }

  return response.json();
}

/**
 * Update user profile
 */
export async function updateUserProfile(
  accessToken: string,
  data: Partial<UserProfile>
): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/users/me`, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error('Failed to update profile');
  }

  return response.json();
}

/**
 * Get user's loan applications
 */
export async function getUserApplications(
  accessToken: string,
  statusFilter?: string
): Promise<UserApplicationsResponse> {
  const url = new URL(`${API_BASE_URL}/users/me/applications`);
  if (statusFilter) {
    url.searchParams.set('status_filter', statusFilter);
  }

  const response = await fetch(url.toString(), {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
    },
  });

  if (!response.ok) {
    throw new Error('Failed to fetch applications');
  }

  return response.json();
}
