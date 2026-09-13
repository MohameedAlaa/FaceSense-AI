import React, { createContext, useState, useEffect, useCallback } from 'react';
import { authApi } from '../lib/api/auth';
import { getAuthToken, setAuthToken, clearAuthToken } from '../lib/api/client';

export const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(getAuthToken());
  const [isLoading, setIsLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  // Restore session from token on mount
  const refreshUser = useCallback(async () => {
    const currentToken = getAuthToken();
    if (!currentToken) {
      setUser(null);
      setToken(null);
      setIsLoading(false);
      return null;
    }

    try {
      const profile = await authApi.me();
      setUser(profile);
      setToken(currentToken);
      setAuthError(null);
      return profile;
    } catch {
      // Token is expired, invalid, or database reset
      clearAuthToken();
      setUser(null);
      setToken(null);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();

    // Listen to unauthorized events from client.js (e.g. 401 response)
    const handleUnauthorized = () => {
      clearAuthToken();
      setUser(null);
      setToken(null);
    };

    window.addEventListener('facesense:unauthorized', handleUnauthorized);
    return () => {
      window.removeEventListener('facesense:unauthorized', handleUnauthorized);
    };
  }, [refreshUser]);

  const login = async (email, password) => {
    setAuthError(null);
    try {
      const data = await authApi.login(email, password);
      setAuthToken(data.access_token);
      setToken(data.access_token);

      // Fetch full user details from /auth/me
      const profile = await authApi.me();
      setUser(profile);
      return profile;
    } catch (err) {
      setAuthError(err.message || 'Login failed');
      throw err;
    }
  };

  const register = async (email, password) => {
    setAuthError(null);
    try {
      return await authApi.register(email, password);
    } catch (err) {
      setAuthError(err.message || 'Registration failed');
      throw err;
    }
  };

  const logout = () => {
    clearAuthToken();
    setUser(null);
    setToken(null);
    setAuthError(null);
  };

  const value = {
    user,
    token,
    isLoading,
    authError,
    isAuthenticated: Boolean(user && token),
    isAdmin: user?.role === 'admin',
    login,
    register,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
