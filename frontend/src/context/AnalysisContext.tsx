import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { QueueItem, HealthResponse, ClassItem, Sex } from '../types';
import { api, ApiError } from '../services/api';

interface AnalysisContextType {
  items: QueueItem[];
  selectedItemId: string | null;
  selectedItem: QueueItem | null;
  isAnalyzing: boolean;
  batchError: string | null;
  health: HealthResponse | null;
  classes: ClassItem[];
  addFiles: (files: File[]) => void;
  removeItem: (id: string) => void;
  replaceFile: (id: string, newFile: File) => void;
  updateMetadata: (id: string, updates: Partial<{ age: string; sex: Sex; localization: string }>) => void;
  applyMetadataToAll: (metadata: { age: string; sex: Sex; localization: string }) => void;
  runBatchAnalysis: () => Promise<boolean>;
  clearSession: () => void;
  selectItem: (id: string) => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [classes, setClasses] = useState<ClassItem[]>([]);

  // Fetch initial health & class lists on mount
  useEffect(() => {
    api.getHealth()
      .then(setHealth)
      .catch((err) => console.warn('Backend health check initial attempt:', err.message));

    api.getClasses()
      .then((res) => setClasses(res.classes))
      .catch((err) => console.warn('Class list fetch initial attempt:', err.message));
  }, []);

  const addFiles = (files: File[]) => {
    const validImageTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    const maxFiles = 15;
    const maxSize = 15 * 1024 * 1024; // 15MB

    const newItems: QueueItem[] = [];

    files.forEach((file) => {
      if (items.length + newItems.length >= maxFiles) {
        return;
      }
      if (!validImageTypes.includes(file.type)) {
        console.warn(`Skipping invalid file format: ${file.name}`);
        return;
      }
      if (file.size > maxSize) {
        console.warn(`Skipping oversized file: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)}MB)`);
        return;
      }

      const id = `${Date.now()}_${Math.random().toString(36).substring(2, 9)}_${file.name}`;
      const previewUrl = URL.createObjectURL(file);

      newItems.push({
        id,
        file,
        previewUrl,
        age: '',
        sex: 'unknown',
        localization: 'unknown',
        status: 'ready',
      });
    });

    if (newItems.length > 0) {
      setItems((prev) => {
        const updated = [...prev, ...newItems];
        if (!selectedItemId) {
          setSelectedItemId(newItems[0].id);
        }
        return updated;
      });
    }
  };

  const removeItem = (id: string) => {
    setItems((prev) => {
      const target = prev.find((i) => i.id === id);
      if (target?.previewUrl) {
        URL.revokeObjectURL(target.previewUrl);
      }
      const updated = prev.filter((i) => i.id !== id);
      if (selectedItemId === id) {
        setSelectedItemId(updated.length > 0 ? updated[0].id : null);
      }
      return updated;
    });
  };

  const replaceFile = (id: string, newFile: File) => {
    setItems((prev) =>
      prev.map((item) => {
        if (item.id === id) {
          if (item.previewUrl) {
            URL.revokeObjectURL(item.previewUrl);
          }
          return {
            ...item,
            file: newFile,
            previewUrl: URL.createObjectURL(newFile),
            status: 'ready',
            result: undefined,
            error: undefined,
          };
        }
        return item;
      })
    );
  };

  const updateMetadata = (
    id: string,
    updates: Partial<{ age: string; sex: Sex; localization: string }>
  ) => {
    setItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...updates } : item))
    );
  };

  const applyMetadataToAll = (metadata: { age: string; sex: Sex; localization: string }) => {
    setItems((prev) =>
      prev.map((item) => ({
        ...item,
        age: metadata.age,
        sex: metadata.sex,
        localization: metadata.localization,
      }))
    );
  };

  const selectItem = (id: string) => {
    setSelectedItemId(id);
  };

  const runBatchAnalysis = async (): Promise<boolean> => {
    if (items.length === 0) return false;

    setIsAnalyzing(true);
    setBatchError(null);

    // Update all items to processing status
    setItems((prev) => prev.map((item) => ({ ...item, status: 'processing' })));

    try {
      const payload = items.map((item) => ({
        id: item.file.name,
        file: item.file,
        age: item.age.trim() !== '' ? parseFloat(item.age) : null,
        sex: item.sex,
        localization: item.localization,
      }));

      const response = await api.predictBatch(payload, true);

      // Match returned results back to items by precise submission index
      setItems((prev) =>
        prev.map((item, idx) => {
          const res = response.results[idx] || response.results.find((r) => r.id === item.file.name);
          if (!res) {
            return { ...item, status: 'failed', error: 'No response returned for this case.' };
          }
          return {
            ...item,
            status: res.status,
            result: res,
            error: res.error,
          };
        })
      );

      setIsAnalyzing(false);
      return true;
    } catch (err: any) {
      const errorMsg = err instanceof ApiError ? err.message : (err?.message || 'Batch analysis failed.');
      setBatchError(errorMsg);
      setItems((prev) =>
        prev.map((item) => ({
          ...item,
          status: 'failed',
          error: errorMsg,
        }))
      );
      setIsAnalyzing(false);
      return false;
    }
  };

  const clearSession = () => {
    items.forEach((item) => {
      if (item.previewUrl) {
        URL.revokeObjectURL(item.previewUrl);
      }
    });
    setItems([]);
    setSelectedItemId(null);
    setBatchError(null);
  };

  const selectedItem = items.find((i) => i.id === selectedItemId) || null;

  return (
    <AnalysisContext.Provider
      value={{
        items,
        selectedItemId,
        selectedItem,
        isAnalyzing,
        batchError,
        health,
        classes,
        addFiles,
        removeItem,
        replaceFile,
        updateMetadata,
        applyMetadataToAll,
        runBatchAnalysis,
        clearSession,
        selectItem,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};
