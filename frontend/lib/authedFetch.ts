/**
 * fetch() wrapper for authenticated API calls.
 *
 * Access tokens live 60 minutes; refresh tokens live 7 days. This wrapper
 * injects the Bearer token and, on a 401, transparently uses the refresh
 * token to mint a new access token and retries the request once — so a
 * long-lived session keeps working without the user noticing. Only when the
 * refresh itself fails (refresh token expired or revoked) is the session
 * genuinely over: auth state is cleared and the user is sent to /login.
 */

import { useAuthStore } from './authStore';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// De-dupes concurrent refreshes: if several requests 401 at once, they all
// await the same single /auth/refresh call instead of stampeding it.
let refreshInFlight: Promise<boolean> | null = null;

async function refreshAccessToken(): Promise<boolean> {
  const { refreshToken } = useAuthStore.getState();
  if (!refreshToken) return false;

  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      try {
        const r = await fetch(`${API_BASE_URL}/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!r.ok) return false;
        const data = await r.json();
        useAuthStore.getState().setTokens(data.access_token, data.refresh_token);
        return true;
      } catch {
        return false;
      } finally {
        refreshInFlight = null;
      }
    })();
  }
  return refreshInFlight;
}

function endSession(): void {
  useAuthStore.getState().logout();
  if (typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
    window.location.href = '/login';
  }
}

export async function authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const send = (): Promise<Response> => {
    const token = useAuthStore.getState().accessToken;
    return fetch(url, {
      ...init,
      headers: {
        ...(init.headers || {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    });
  };

  let res = await send();
  if (res.status !== 401) return res;

  // Access token rejected — try to refresh and retry the request once.
  if (await refreshAccessToken()) {
    res = await send();
    if (res.status !== 401) return res;
  }

  // Refresh unavailable or also rejected — the session is truly over.
  endSession();
  return res;
}
