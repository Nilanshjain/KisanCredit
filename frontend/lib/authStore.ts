/**
 * Authentication state management using Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { UserProfile } from './authApi';

interface AuthState {
  // State
  accessToken: string | null;
  refreshToken: string | null;
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  setTokens: (accessToken: string, refreshToken: string) => void;
  setUser: (user: UserProfile) => void;
  logout: () => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      // Initial state
      accessToken: null,
      refreshToken: null,
      user: null,
      isAuthenticated: false,
      isLoading: false,

      // Set tokens after login
      setTokens: (accessToken, refreshToken) =>
        set({
          accessToken,
          refreshToken,
          isAuthenticated: true,
        }),

      // Set user profile
      setUser: (user) =>
        set({
          user,
        }),

      // Logout and clear state
      logout: () =>
        set({
          accessToken: null,
          refreshToken: null,
          user: null,
          isAuthenticated: false,
        }),

      // Set loading state
      setLoading: (loading) =>
        set({
          isLoading: loading,
        }),
    }),
    {
      name: 'kisan-credit-auth', // LocalStorage key
      partialize: (state) => ({
        // Only persist these fields
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
