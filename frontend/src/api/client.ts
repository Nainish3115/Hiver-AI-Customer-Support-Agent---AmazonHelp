import {
  SupportProcessRequest,
  SupportProcessResponse,
  HealthResponse,
  ReadyResponse,
  StatsResponse,
  RetrievalSearchResponse,
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export class ApiError extends Error {
  status: number;
  detail?: string;
  requestId?: string;

  constructor(message: string, status: number, detail?: string, requestId?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.requestId = requestId;
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
    const reqId = res.headers.get('X-Request-ID') || undefined;
    if (!res.ok) {
      throw new ApiError(`Health check failed with status ${res.status}`, res.status, undefined, reqId);
    }
    const data = await res.json();
    return { ...data, request_id: data.request_id || reqId };
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Backend server unavailable. Please make sure FastAPI is running on port 8000.', 503);
  }
}

export async function checkReady(): Promise<ReadyResponse> {
  try {
    const res = await fetch(`${API_BASE}/ready`, { method: 'GET' });
    const reqId = res.headers.get('X-Request-ID') || undefined;
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      const msg = errJson.message || errJson.error?.message || `Readiness check failed (${res.status})`;
      throw new ApiError(msg, res.status, errJson.detail, reqId);
    }
    const data = await res.json();
    return { ...data, request_id: data.request_id || reqId };
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Readiness probe failed. Pipeline initializing or backend unavailable.', 503);
  }
}

export async function processSupportMessage(req: SupportProcessRequest): Promise<SupportProcessResponse> {
  try {
    const res = await fetch(`${API_BASE}/support/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req),
    });

    const reqId = res.headers.get('X-Request-ID') || undefined;

    if (!res.ok) {
      const errorJson = await res.json().catch(() => ({}));
      const msg = errorJson.message || errorJson.error?.message || errorJson.detail || `Server error (${res.status})`;
      throw new ApiError(msg, res.status, errorJson.detail, reqId);
    }

    const data = await res.json();
    return { ...data, request_id: data.request_id || reqId };
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Failed to process inquiry with backend server.', 500);
  }
}

export async function fetchProjectStats(): Promise<StatsResponse> {
  try {
    const res = await fetch(`${API_BASE}/stats`, { method: 'GET' });
    const reqId = res.headers.get('X-Request-ID') || undefined;
    if (!res.ok) {
      throw new ApiError(`Failed to fetch project stats (${res.status})`, res.status, undefined, reqId);
    }
    const data = await res.json();
    return { ...data, request_id: data.request_id || reqId };
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Failed to load project metrics.', 500);
  }
}

export async function searchRetrievalCorpus(query: string, topK: number = 10): Promise<RetrievalSearchResponse> {
  try {
    const res = await fetch(`${API_BASE}/retrieval/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK }),
    });
    const reqId = res.headers.get('X-Request-ID') || undefined;

    if (!res.ok) {
      const errorJson = await res.json().catch(() => ({}));
      const msg = errorJson.message || errorJson.error?.message || `Retrieval search failed (${res.status})`;
      throw new ApiError(msg, res.status, errorJson.detail, reqId);
    }
    const data = await res.json();
    return { ...data, request_id: data.request_id || reqId };
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Failed to execute retrieval search.', 500);
  }
}
