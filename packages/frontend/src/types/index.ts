export type JobStatus = "pending" | "processing" | "completed" | "failed" | "cancelled";

export interface Job {
  id: string;
  name: string;
  type: "file" | "uuid";
  input: string; // filename or UUID
  status: JobStatus;
  progress: number;
  currentStage?: string;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  duration?: number;
  result?: TranscriptionResult;
  error?: string;
}

export interface TranscriptionResult {
  text: string;
  segments: Array<{
    start: number;
    end: number;
    text: string;
  }>;
  metadata: {
    duration: number;
    language: string;
    model: string;
  };
}

export interface LogEntry {
  timestamp: Date;
  level: "info" | "warning" | "error" | "debug";
  message: string;
  ansi?: string; // Raw ANSI string
}

export interface AnalyticsData {
  totalJobs: number;
  successRate: number;
  avgDuration: number;
  jobsPerDay: Array<{ date: string; count: number }>;
  statusDistribution: Array<{ status: string; count: number }>;
}
