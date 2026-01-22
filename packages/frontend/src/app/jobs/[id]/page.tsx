"use client";

import { useState, useEffect, useMemo, use } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { JobDetails } from "@/components/JobDetails";
import { LogViewer } from "@/components/LogViewer";
import { api } from "@/lib/api";
import { useLogStream } from "@/lib/sse";
import { Job, LogEntry } from "@/types";

interface JobPageProps {
  params: Promise<{ id: string }>;
}

export default function JobPage({ params }: JobPageProps) {
  const { id } = use(params);
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // SSE stream for real-time updates (only for active jobs)
  const isActiveJob = job?.status === "processing" || job?.status === "pending";
  const { logs: streamingLogs, isConnected, jobState } = useLogStream(isActiveJob ? id : null);

  // Convert saved logs from database to LogEntry format
  const savedLogs = useMemo((): LogEntry[] => {
    if (!job?.logs) return [];
    return job.logs
      .filter((log) => log.type === "log") // Only show actual log entries, not progress/status
      .map((log) => ({
        timestamp: new Date(log.timestamp),
        level: (log.level || "info") as LogEntry["level"],
        message: log.message,
        ansi: log.ansi || log.message,
      }));
  }, [job?.logs]);

  // Use streaming logs for active jobs, saved logs for completed jobs
  const displayLogs = isActiveJob ? streamingLogs : savedLogs;
  const hasLogs = displayLogs.length > 0;

  // Update job from SSE events
  useEffect(() => {
    if (job && jobState.status && jobState.status !== job.status) {
      setJob((prev) =>
        prev
          ? {
              ...prev,
              status: jobState.status as Job["status"],
              progress: jobState.progress,
              currentStage: jobState.stage ?? undefined,
              currentStep: jobState.step ?? undefined,
              totalSteps: jobState.totalSteps ?? undefined,
              error: jobState.error ?? undefined,
            }
          : prev
      );
    }
    // Also update progress even if status hasn't changed
    if (job && (jobState.progress !== job.progress || jobState.step !== job.currentStep)) {
      setJob((prev) =>
        prev
          ? {
              ...prev,
              progress: jobState.progress,
              currentStage: jobState.stage ?? prev.currentStage,
              currentStep: jobState.step ?? prev.currentStep,
              totalSteps: jobState.totalSteps ?? prev.totalSteps,
            }
          : prev
      );
    }
  }, [jobState.status, jobState.progress, jobState.stage, jobState.step, jobState.totalSteps, jobState.error]);

  // Refetch when job completes to get final data (result, duration, etc.)
  useEffect(() => {
    if (jobState.isComplete) {
      fetchJob();
    }
  }, [jobState.isComplete]);

  const fetchJob = async () => {
    try {
      const jobData = await api.getJob(id);
      if (!jobData) {
        setError("Job not found");
        return;
      }
      setJob(jobData);
    } catch (err) {
      setError("Failed to fetch job");
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch only - no more polling!
  useEffect(() => {
    fetchJob();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="container mx-auto py-8 px-4">
        <Button variant="ghost" onClick={() => router.push("/")}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Button>
        <div className="mt-8 text-center">
          <h1 className="text-2xl font-bold text-red-600">{error || "Job not found"}</h1>
          <p className="text-muted-foreground mt-2">
            The job you're looking for doesn't exist or has been deleted.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4 space-y-8">
        {/* Header */}
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => router.push("/")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div>
            <h1 className="text-2xl font-bold">Job Details</h1>
            <p className="text-muted-foreground">{job.name}</p>
          </div>
        </div>

        {/* Job Details */}
        <JobDetails job={job} />

        {/* Log Viewer - show for active jobs (streaming) or completed jobs (saved) */}
        {(isActiveJob || hasLogs) && (
          <LogViewer
            jobId={job.id}
            logs={displayLogs}
            isConnected={isActiveJob ? isConnected : false}
            isHistorical={!isActiveJob}
          />
        )}
      </div>
    </div>
  );
}
