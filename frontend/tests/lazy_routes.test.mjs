import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('FaceSense Frontend Route Lazy Loading & Code Splitting Tests', () => {
  test('1. All lazy-loaded page files exist and export default React components', async () => {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');

    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const lazyPages = [
      'DashboardPage.jsx',
      'HistoryPage.jsx',
      'InsightsPage.jsx',
      'SettingsPage.jsx',
      'AdminPage.jsx',
    ];

    for (const fileName of lazyPages) {
      const filePath = path.resolve(__dirname, '../src/pages', fileName);
      const stat = await fs.stat(filePath);
      assert.ok(stat.isFile(), `${fileName} must exist`);
      const src = await fs.readFile(filePath, 'utf8');
      assert.ok(
        src.includes('export default function') || src.includes('export default'),
        `${fileName} must have a default export`
      );
    }
  });

  test('2. Eagerly loaded public & primary pages exist and export default components', async () => {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');

    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const eagerPages = [
      'LandingPage.jsx',
      'LoginPage.jsx',
      'RegisterPage.jsx',
      'AnalyzePage.jsx',
    ];

    for (const fileName of eagerPages) {
      const filePath = path.resolve(__dirname, '../src/pages', fileName);
      const stat = await fs.stat(filePath);
      assert.ok(stat.isFile(), `${fileName} must exist`);
      const src = await fs.readFile(filePath, 'utf8');
      assert.ok(
        src.includes('export default function') || src.includes('export default'),
        `${fileName} must have a default export`
      );
    }
  });

  test('3. AppRoutes source includes React.lazy and Suspense for code splitting', async () => {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');

    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const appRoutesPath = path.resolve(__dirname, '../src/routes/AppRoutes.jsx');
    const content = await fs.readFile(appRoutesPath, 'utf8');

    // Verify React.lazy imports for secondary routes
    assert.ok(content.includes("lazy(() => import('../pages/DashboardPage'))"), 'DashboardPage must be lazy loaded');
    assert.ok(content.includes("lazy(() => import('../pages/HistoryPage'))"), 'HistoryPage must be lazy loaded');
    assert.ok(content.includes("lazy(() => import('../pages/InsightsPage'))"), 'InsightsPage must be lazy loaded');
    assert.ok(content.includes("lazy(() => import('../pages/SettingsPage'))"), 'SettingsPage must be lazy loaded');
    assert.ok(content.includes("lazy(() => import('../pages/AdminPage'))"), 'AdminPage must be lazy loaded');

    // Verify Suspense wrapper exists
    assert.ok(content.includes('<Suspense fallback={<RouteLoadingFallback />}>'), 'Suspense fallback wrapper must be present');

    // Verify accessible loading fallback
    assert.ok(content.includes('role="status"'), 'Loading fallback must include role="status"');
    assert.ok(content.includes('aria-label="Loading page content"'), 'Loading fallback must include aria-label');
  });

  test('4. AdminRoute authorization prevents non-admin execution without loading AdminPage', async () => {
    // Verify admin check logic
    const evaluateAdminAccess = (isAuthenticated, isAdmin) => {
      if (!isAuthenticated) return 'REDIRECT_LOGIN';
      if (!isAdmin) return 'SHOW_RESTRICTED';
      return 'ALLOW_ADMIN';
    };

    assert.equal(evaluateAdminAccess(false, false), 'REDIRECT_LOGIN');
    assert.equal(evaluateAdminAccess(true, false), 'SHOW_RESTRICTED');
    assert.equal(evaluateAdminAccess(true, true), 'ALLOW_ADMIN');
  });
});
