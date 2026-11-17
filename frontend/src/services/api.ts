import {
  HealthResponse,
  ClassesResponse,
  PredictResponse,
  BatchResponse,
  ExplainResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string'
          ? errorJson.detail
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Non-JSON error body
    }
    throw new ApiError(errorDetail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    return handleResponse<HealthResponse>(res);
  },

  async getClasses(): Promise<ClassesResponse> {
    const res = await fetch(`${API_BASE_URL}/api/classes`);
    return handleResponse<ClassesResponse>(res);
  },

  async predictSingle(
    file: File,
    metadata?: { age?: number | null; sex?: string | null; localization?: string | null },
    includeGradcam: boolean = true
  ): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append('image', file);
    if (metadata?.age !== undefined && metadata.age !== null) {
      formData.append('age', String(metadata.age));
    }
    if (metadata?.sex) {
      formData.append('sex', metadata.sex);
    }
    if (metadata?.localization) {
      formData.append('localization', metadata.localization);
    }
    formData.append('include_gradcam', String(includeGradcam));

    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<PredictResponse>(res);
  },

  async predictBatch(
    items: Array<{
      id: string;
      file: File;
      age?: number | null;
      sex?: string | null;
      localization?: string | null;
    }>,
    includeGradcam: boolean = true
  ): Promise<BatchResponse> {
    const formData = new FormData();

    // Map metadata as an array matching image order precisely
    const metadataList = items.map((item) => ({
      age: item.age !== undefined && item.age !== null ? item.age : null,
      sex: item.sex || 'unknown',
      localization: item.localization || 'unknown',
    }));

    items.forEach((item) => {
      formData.append('images', item.file, item.file.name);
    });

    formData.append('metadata_json', JSON.stringify(metadataList));
    formData.append('include_gradcam', String(includeGradcam));

    const res = await fetch(`${API_BASE_URL}/api/predict/batch`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<BatchResponse>(res);
  },

  async explain(
    file: File,
    metadata?: {
      age?: number | null;
      sex?: string | null;
      localization?: string | null;
      targetClass?: string | null;
    }
  ): Promise<ExplainResponse> {
    const formData = new FormData();
    formData.append('image', file);
    if (metadata?.age !== undefined && metadata.age !== null) {
      formData.append('age', String(metadata.age));
    }
    if (metadata?.sex) {
      formData.append('sex', metadata.sex);
    }
    if (metadata?.localization) {
      formData.append('localization', metadata.localization);
    }
    if (metadata?.targetClass) {
      formData.append('target_class', metadata.targetClass);
    }

    const res = await fetch(`${API_BASE_URL}/api/explain`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<ExplainResponse>(res);
  },
};
