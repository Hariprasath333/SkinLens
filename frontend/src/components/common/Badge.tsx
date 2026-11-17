import React from 'react';

interface BadgeProps {
  category: 'Benign' | 'Malignant' | 'Precancerous / In-situ' | string;
  size?: 'sm' | 'md';
}

export const CategoryBadge: React.FC<BadgeProps> = ({ category, size = 'md' }) => {
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';

  if (category === 'Malignant') {
    return (
      <span
        className={`inline-flex items-center rounded-full font-medium border border-rose-200 bg-rose-50 text-rose-800 ${sizeClasses}`}
      >
        <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-rose-500" />
        Malignant
      </span>
    );
  }

  if (category === 'Precancerous / In-situ') {
    return (
      <span
        className={`inline-flex items-center rounded-full font-medium border border-amber-200 bg-amber-50 text-amber-800 ${sizeClasses}`}
      >
        <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-amber-500" />
        Precancerous
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium border border-emerald-200 bg-emerald-50 text-emerald-800 ${sizeClasses}`}
    >
      <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-emerald-500" />
      Benign
    </span>
  );
};
