import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';
import { THEME_STORAGE_KEY, resolveTheme, applyThemeToDocument, getSystemTheme } from '../lib/theme';

export { THEME_STORAGE_KEY, resolveTheme, applyThemeToDocument, getSystemTheme };

const ThemeContext = createContext(null);

/**
 * ThemeProvider implements:
 * 1. User preference: 'light' | 'dark' | 'system'
 * 2. Resolved theme: 'light' | 'dark'
 * 3. Immediate DOM application to document.documentElement (.dark class and color-scheme)
 * 4. Persistence in localStorage ('facesense_theme')
 * 5. Dynamic updates on system prefers-color-scheme changes when preference is 'system'
 */
export function ThemeProvider({ children }) {
  const [preference, setPreference] = useState(() => {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (saved === 'light' || saved === 'dark' || saved === 'system') {
        return saved;
      }
    } catch {
      // Ignore localStorage errors
    }
    return 'system';
  });

  const [resolvedTheme, setResolvedTheme] = useState(() => {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (saved === 'light' || saved === 'dark' || saved === 'system') {
        return resolveTheme(saved);
      }
    } catch {
      // Ignore localStorage errors
    }
    return resolveTheme('system');
  });

  // Keep DOM and state synchronized with preference and OS settings
  useEffect(() => {
    const currentResolved = resolveTheme(preference);
    setResolvedTheme(currentResolved);
    applyThemeToDocument(currentResolved);

    if (typeof window === 'undefined' || !window.matchMedia) return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleMediaChange = (e) => {
      if (preference === 'system') {
        const nextResolved = e.matches ? 'dark' : 'light';
        setResolvedTheme(nextResolved);
        applyThemeToDocument(nextResolved);
      }
    };

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleMediaChange);
      return () => mediaQuery.removeEventListener('change', handleMediaChange);
    } else if (mediaQuery.addListener) {
      mediaQuery.addListener(handleMediaChange);
      return () => mediaQuery.removeListener(handleMediaChange);
    }
  }, [preference]);

  const setTheme = useCallback((mode) => {
    if (mode !== 'light' && mode !== 'dark' && mode !== 'system') return;
    const nextResolved = resolveTheme(mode);
    setPreference(mode);
    setResolvedTheme(nextResolved);
    applyThemeToDocument(nextResolved);

    try {
      localStorage.setItem(THEME_STORAGE_KEY, mode);
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  const value = useMemo(() => ({
    preference,
    themeMode: preference, // backward-compatible alias
    resolvedTheme,
    effectiveTheme: resolvedTheme, // backward-compatible alias
    setTheme,
  }), [preference, resolvedTheme, setTheme]);

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
