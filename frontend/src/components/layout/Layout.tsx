import React, { ReactNode } from 'react';
import { Navbar } from './Navbar';
import { Footer } from './Footer';

export const Layout: React.FC<{ children: ReactNode }> = ({ children }) => {
  return (
    <div className="flex min-h-screen flex-col bg-background font-sans text-zinc-900">
      <Navbar />
      <main className="flex-1">{children}</main>
      <Footer />
    </div>
  );
};
