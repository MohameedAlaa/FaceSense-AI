import React, { useState } from 'react';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import MobileNav from './MobileNav';

export default function AppShell({ children }) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <div className="h-screen w-full max-w-full overflow-hidden flex flex-row bg-[#F7F7FC] dark:bg-[#0B1020] text-slate-900 dark:text-[#F8FAFC] antialiased">
      {/* Desktop Persistent Sidebar */}
      <div className="hidden lg:block h-full shrink-0">
        <Sidebar />
      </div>

      {/* Mobile Drawer Overlay Sidebar */}
      {mobileNavOpen && (
        <div 
          className="lg:hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex animate-fade-in"
          onClick={() => setMobileNavOpen(false)}
        >
          <div 
            className="w-72 h-full bg-white dark:bg-[#12182B] shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <Sidebar onCloseMobile={() => setMobileNavOpen(false)} />
          </div>
        </div>
      )}

      {/* Main Content Workspace */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <TopBar onToggleMobileNav={() => setMobileNavOpen((prev) => !prev)} />
        
        <main className="flex-1 overflow-y-auto overflow-x-hidden pb-16 lg:pb-0 bg-[#F7F7FC] dark:bg-[#0B1020] transition-colors min-w-0 max-w-full">
          {children}
        </main>
      </div>

      {/* Mobile Bottom Navigation Bar */}
      <MobileNav />
    </div>
  );
}
