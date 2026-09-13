/**
 * FaceSense Theme Management Utilities
 * Handles user preference ('light' | 'dark' | 'system') and theme resolution ('light' | 'dark').
 */

export const THEME_STORAGE_KEY = 'facesense_theme';

/**
 * Detects current operating system color scheme preference.
 * @returns {'light' | 'dark'}
 */
export const getSystemTheme = () => {
  if (typeof window === 'undefined' || !window.matchMedia) return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

/**
 * Resolves effective theme given a user preference.
 * @param {'light' | 'dark' | 'system' | null} preference
 * @returns {'light' | 'dark'}
 */
export const resolveTheme = (preference) => {
  if (preference === 'system' || !preference) {
    return getSystemTheme();
  }
  return preference === 'dark' ? 'dark' : 'light';
};

/**
 * Synchronously applies resolved theme to document root.
 * Sets .dark / .light class, style.colorScheme, and data-theme attribute.
 * @param {'light' | 'dark'} resolved
 */
export const applyThemeToDocument = (resolved) => {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  if (resolved === 'dark') {
    root.classList.add('dark');
    root.classList.remove('light');
    root.style.colorScheme = 'dark';
    root.setAttribute('data-theme', 'dark');
  } else {
    root.classList.add('light');
    root.classList.remove('dark');
    root.style.colorScheme = 'light';
    root.setAttribute('data-theme', 'light');
  }
};
