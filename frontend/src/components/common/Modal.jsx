import React, { useEffect } from 'react';

/**
 * Accessible Modal dialog component
 */
export default function Modal({
  isOpen,
  onClose,
  title,
  children,
  maxWidth = 'max-w-md',
}) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <div 
        className={`w-full ${maxWidth} bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-2xl p-6 shadow-2xl relative text-left`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100 dark:border-[#1E294B]">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">{title}</h3>
          <button 
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-600 dark:hover:text-white transition-colors"
            title="Close"
          >
            <span className="material-symbols-outlined text-[18px]">close</span>
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
