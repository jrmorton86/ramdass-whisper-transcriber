import { Job, AnalyticsData } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

// API response types
interface JobListResponse {
  jobs: Job[];
  total: number;
}

export const api = {
  // Get all jobs
  getJobs: async (): Promise<Job[]> => {
    const response = await fetch(`${API_URL}/api/jobs`);
    const data = await handleResponse<JobListResponse>(response);
    // Convert date strings to Date objects
    return data.jobs.map(job => ({
      ...job,
      createdAt: new Date(job.createdAt),
      startedAt: job.startedAt ? new Date(job.startedAt) : undefined,
      completedAt: job.completedAt ? new Date(job.completedAt) : undefined,
    }));
  },

  // Get single job
  getJob: async (id: string): Promise<Job | null> => {
    const response = await fetch(`${API_URL}/api/jobs/${id}`);
    if (response.status === 404) return null;
    const data = await handleResponse<Job>(response);
    return {
      ...data,
      createdAt: new Date(data.createdAt),
      startedAt: data.startedAt ? new Date(data.startedAt) : undefined,
      completedAt: data.completedAt ? new Date(data.completedAt) : undefined,
    };
  },

  // Create new job from UUID
  createJob: async (data: { type: "file" | "uuid"; input: string }): Promise<Job> => {
    const response = await fetch(`${API_URL}/api/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: data.type === "uuid" ? `UUID: ${data.input}` : data.input,
        type: data.type,
        input: data.input,
      }),
    });
    const job = await handleResponse<Job>(response);
    return {
      ...job,
      createdAt: new Date(job.createdAt),
      startedAt: job.startedAt ? new Date(job.startedAt) : undefined,
      completedAt: job.completedAt ? new Date(job.completedAt) : undefined,
    };
  },

  // Upload file and create job
  uploadFile: async (file: File): Promise<Job> => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/api/upload`, {
      method: "POST",
      body: formData,
    });
    const job = await handleResponse<Job>(response);
    return {
      ...job,
      createdAt: new Date(job.createdAt),
      startedAt: job.startedAt ? new Date(job.startedAt) : undefined,
      completedAt: job.completedAt ? new Date(job.completedAt) : undefined,
    };
  },

  // Delete job
  deleteJob: async (id: string): Promise<boolean> => {
    const response = await fetch(`${API_URL}/api/jobs/${id}`, {
      method: "DELETE",
    });
    return response.ok;
  },

  // Cancel job
  cancelJob: async (id: string): Promise<boolean> => {
    const response = await fetch(`${API_URL}/api/jobs/${id}/cancel`, {
      method: "POST",
    });
    return response.ok;
  },

  // Get analytics
  getAnalytics: async (): Promise<AnalyticsData> => {
    const response = await fetch(`${API_URL}/api/analytics`);
    return handleResponse<AnalyticsData>(response);
  },
};

// SSE log streaming URL
export function getLogsStreamUrl(jobId: string): string {
  return `${API_URL}/api/jobs/${jobId}/logs`;
}
