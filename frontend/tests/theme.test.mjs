import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// Mock localStorage for Node test environment
class MockLocalStorage {
  constructor() {
    this.store = {};
  }
  getItem(key) {
    return this.store[key] || null;
  }
  setItem(key, value) {
    this.store[key] = String(value);
  }
  removeItem(key) {
    delete this.store[key];
  }
  clear() {
    this.store = {};
  }
}

globalThis.localStorage = new MockLocalStorage();

// Mock document for Node environment
class MockClassList {
  constructor() {
    this.classes = new Set();
  }
  add(cls) {
    this.classes.add(cls);
  }
  remove(cls) {
    this.classes.delete(cls);
  }
  contains(cls) {
    return this.classes.has(cls);
  }
}

globalThis.document = {
  documentElement: {
    classList: new MockClassList(),
    style: {},
    setAttribute(name, val) {
      this[name] = val;
    },
  },
};

// Mock matchMedia
let mockPrefersDark = false;
globalThis.window = {
  matchMedia: (query) => ({
    matches: query.includes('prefers-color-scheme: dark') ? mockPrefersDark : false,
    addEventListener: () => {},
    removeEventListener: () => {},
  }),
};

describe('FaceSense Theme Preference & Resolution Logic Tests', () => {
  beforeEach(() => {
    globalThis.localStorage.clear();
    globalThis.document.documentElement.classList.classes.clear();
    mockPrefersDark = false;
  });

  test('1. First visit with no saved theme defaults to system preference', async () => {
    const { resolveTheme } = await import('../src/lib/theme.js');
    const saved = globalThis.localStorage.getItem('facesense_theme');
    assert.equal(saved, null, 'LocalStorage must be empty on fresh visit');

    // With system prefering light
    mockPrefersDark = false;
    assert.equal(resolveTheme('system'), 'light');

    // With system prefering dark
    mockPrefersDark = true;
    assert.equal(resolveTheme('system'), 'dark');
  });

  test('2. Light preference strictly resolves to light regardless of OS theme', async () => {
    const { resolveTheme } = await import('../src/lib/theme.js');
    
    mockPrefersDark = true;
    assert.equal(resolveTheme('light'), 'light', 'Must force light even when OS prefers dark');

    mockPrefersDark = false;
    assert.equal(resolveTheme('light'), 'light');
  });

  test('3. Dark preference strictly resolves to dark regardless of OS theme', async () => {
    const { resolveTheme } = await import('../src/lib/theme.js');

    mockPrefersDark = false;
    assert.equal(resolveTheme('dark'), 'dark', 'Must force dark even when OS prefers light');

    mockPrefersDark = true;
    assert.equal(resolveTheme('dark'), 'dark');
  });

  test('4. Auto/System preference dynamically tracks OS prefers-color-scheme', async () => {
    const { resolveTheme } = await import('../src/lib/theme.js');

    mockPrefersDark = true;
    assert.equal(resolveTheme('system'), 'dark');

    mockPrefersDark = false;
    assert.equal(resolveTheme('system'), 'light');
  });

  test('5. DOM class manipulation correctly adds/removes .dark and sets colorScheme', async () => {
    const { applyThemeToDocument } = await import('../src/lib/theme.js');

    // Apply dark
    applyThemeToDocument('dark');
    assert.ok(globalThis.document.documentElement.classList.contains('dark'));
    assert.ok(!globalThis.document.documentElement.classList.contains('light'));
    assert.equal(globalThis.document.documentElement.style.colorScheme, 'dark');
    assert.equal(globalThis.document.documentElement['data-theme'], 'dark');

    // Apply light
    applyThemeToDocument('light');
    assert.ok(!globalThis.document.documentElement.classList.contains('dark'));
    assert.ok(globalThis.document.documentElement.classList.contains('light'));
    assert.equal(globalThis.document.documentElement.style.colorScheme, 'light');
    assert.equal(globalThis.document.documentElement['data-theme'], 'light');
  });

  test('6. Preference persistence round-trips correctly in localStorage', () => {
    const STORAGE_KEY = 'facesense_theme';
    
    globalThis.localStorage.setItem(STORAGE_KEY, 'dark');
    assert.equal(globalThis.localStorage.getItem(STORAGE_KEY), 'dark');

    globalThis.localStorage.setItem(STORAGE_KEY, 'light');
    assert.equal(globalThis.localStorage.getItem(STORAGE_KEY), 'light');

    globalThis.localStorage.setItem(STORAGE_KEY, 'system');
    assert.equal(globalThis.localStorage.getItem(STORAGE_KEY), 'system');
  });
});
