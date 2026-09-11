import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [themeMode, setThemeMode] = useState(() => {
    try {
      const saved = localStorage.getItem('facesense_theme');
      if (saved === 'light' || saved === 'dark' || saved === 'system') {
        return saved;
      }
    } catch {
      // Ignore localStorage errors
    }
    return 'system';
  });

  const [effectiveTheme, setEffectiveTheme] = useState('dark');

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const applyTheme = () => {
      let resolved = themeMode;
      if (themeMode === 'system') {
        resolved = mediaQuery.matches ? 'dark' : 'light';
      }

      setEffectiveTheme(resolved);

      const root = document.documentElement;
      if (resolved === 'dark') {
        root.classList.add('dark');
        root.classList.remove('light');
        root.style.colorScheme = 'dark';
      } else {
        root.classList.add('light');
        root.classList.remove('dark');
        root.style.colorScheme = 'light';
      }
    };

    applyTheme();

    const handler = () => {
      if (themeMode === 'system') {
        applyTheme();
      }
    };

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handler);
      return () => mediaQuery.removeEventListener('change', handler);
    } else {
      mediaQuery.addListener(handler);
      return () => mediaQuery.removeListener(handler);
    }
  }, [themeMode]);

  const setTheme = (mode) => {
    if (mode !== 'light' && mode !== 'dark' && mode !== 'system') return;
    setThemeMode(mode);
    try {
      localStorage.setItem('facesense_theme', mode);
    } catch {
      // Ignore localStorage errors
    }
  };

  return (
    <ThemeContext.Provider value={{ themeMode, effectiveTheme, setTheme }}>
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
