import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AnalysisProvider } from './context/AnalysisContext';
import { Layout } from './components/layout/Layout';
import { HomePage } from './pages/HomePage';
import { AnalyzePage } from './pages/AnalyzePage';
import { ResultsPage } from './pages/ResultsPage';
import { MethodologyPage } from './pages/MethodologyPage';

export const App: React.FC = () => {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/methodology" element={<MethodologyPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </AnalysisProvider>
  );
};

export default App;
