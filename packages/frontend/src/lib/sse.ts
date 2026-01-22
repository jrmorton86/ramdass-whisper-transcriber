"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { LogEntry } from "@/types";
import { getLogsStreamUrl } from "./api";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UseLogStreamOptions {
  maxLogs?: number;
  autoReconnect?: boolean;
  reconnectDelay?: number;
}

interface JobStreamState {
  status: string | null;
  progress: number;
  stage: string | null;
  step: number | null;
  totalSteps: number | null;
  error: string | null;
  isComplete: boolean;
}

interface UseLogStreamReturn {
  logs: LogEntry[];
  isConnected: boolean;
  error: string | null;
  jobState: JobStreamState;
  connect: () => void;
  disconnect: () => void;
  clearLogs: () => void;
}

export function useLogStream(
  jobId: string | null,
  options: UseLogStreamOptions = {}
): UseLogStreamReturn {
  const { maxLogs = 1000, autoReconnect = true, reconnectDelay = 3000 } = options;

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobStreamState>({
    status: null,
    progress: 0,
    stage: null,
    step: null,
    totalSteps: null,
    error: null,
    isComplete: false,
  });

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastLoggedStepRef = useRef<number | null>(null);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsConnected(false);
  }, []);

  // Helper to add a log entry with deduplication and max limit
  const addLogEntry = useCallback((entry: LogEntry) => {
    setLogs((prev) => {
      // Deduplicate by timestamp + message
      const key = `${entry.timestamp.getTime()}-${entry.message}`;
      const exists = prev.some(
        (l) => `${l.timestamp.getTime()}-${l.message}` === key
      );
      if (exists) return prev;

      const newLogs = [...prev, entry];
      if (newLogs.length > maxLogs) {
        return newLogs.slice(-maxLogs);
      }
      return newLogs;
    });
  }, [maxLogs]);

  const connect = useCallback(async () => {
    if (!jobId) return;

    disconnect();
    setError(null);

    // First, fetch historical logs to catch any emitted before connection
    try {
      const historyResponse = await fetch(
        `${API_URL}/api/jobs/${jobId}/logs/history`
      );
      if (historyResponse.ok) {
        const { logs: historyLogs } = await historyResponse.json();
        if (historyLogs && historyLogs.length > 0) {
          // Add historical logs
          for (const log of historyLogs) {
            if (log.type === "log") {
              addLogEntry({
                timestamp: new Date(log.timestamp),
                level: log.level || "info",
                message: log.message,
                ansi: log.ansi || log.message,
              });
            }
          }
        }
      }
    } catch (e) {
      console.error("Failed to fetch log history:", e);
    }

    // Now connect to SSE stream
    const url = getLogsStreamUrl(jobId);
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    // Handle named events - backend sends "log", "progress", "status", "complete", "error", "ping"
    const handleLogEvent = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        addLogEntry({
          timestamp: new Date(data.timestamp),
          level: data.level || "info",
          message: data.message,
          ansi: data.ansi || data.message,
        });
      } catch (e) {
        console.error("Failed to parse log event:", e);
      }
    };

    const handleProgressEvent = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        // Update job state with progress
        setJobState((prev) => ({
          ...prev,
          progress: data.progress ?? prev.progress,
          stage: data.stage ?? prev.stage,
          step: data.step ?? prev.step,
          totalSteps: data.totalSteps ?? prev.totalSteps,
        }));
        // Add step change as a log entry (only when step changes, not every progress update)
        if (data.stage && data.step && data.step !== lastLoggedStepRef.current) {
          lastLoggedStepRef.current = data.step;
          const stepInfo = data.totalSteps
            ? `[Step ${data.step}/${data.totalSteps}]`
            : `[Step ${data.step}]`;
          addLogEntry({
            timestamp: new Date(data.timestamp),
            level: "info",
            message: `${stepInfo} ${data.stage}`,
            ansi: `${stepInfo} ${data.stage}`,
          });
        }
      } catch (e) {
        console.error("Failed to parse progress event:", e);
      }
    };

    const handleStatusEvent = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        // Update job state with status
        const isComplete = ["completed", "failed", "cancelled"].includes(data.status);
        setJobState((prev) => ({
          ...prev,
          status: data.status,
          error: data.error ?? prev.error,
          isComplete,
        }));
        addLogEntry({
          timestamp: new Date(data.timestamp),
          level: data.status === "failed" ? "error" : "info",
          message: `[Status] ${data.status}${data.error ? `: ${data.error}` : ""}`,
          ansi: `[Status] ${data.status}${data.error ? `: ${data.error}` : ""}`,
        });
      } catch (e) {
        console.error("Failed to parse status event:", e);
      }
    };

    const handleCompleteEvent = () => {
      // Job finished, close connection
      setJobState((prev) => ({ ...prev, isComplete: true }));
      eventSource.close();
      setIsConnected(false);
    };

    const handleErrorEvent = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        addLogEntry({
          timestamp: new Date(),
          level: "error",
          message: data.message || "Unknown error",
          ansi: data.message || "Unknown error",
        });
      } catch (e) {
        // Ignore parse errors for error events
      }
      eventSource.close();
      setIsConnected(false);
    };

    // Register named event listeners
    eventSource.addEventListener("log", handleLogEvent);
    eventSource.addEventListener("progress", handleProgressEvent);
    eventSource.addEventListener("status", handleStatusEvent);
    eventSource.addEventListener("complete", handleCompleteEvent);
    eventSource.addEventListener("error", handleErrorEvent);
    // ping events are just keepalives, no need to handle

    eventSource.onerror = () => {
      eventSource.close();
      setIsConnected(false);
      setError("Connection lost");

      if (autoReconnect) {
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectDelay);
      }
    };
  }, [jobId, maxLogs, autoReconnect, reconnectDelay, disconnect, addLogEntry]);

  // Auto-connect when jobId changes
  useEffect(() => {
    if (jobId) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [jobId, connect, disconnect]);

  return {
    logs,
    isConnected,
    error,
    jobState,
    connect,
    disconnect,
    clearLogs,
  };
}

// ============================================================================
// Dashboard SSE Stream Hook
// ============================================================================

interface DashboardEvent {
  type: "job_created" | "job_updated" | "job_progress";
  timestamp: string;
  jobId: string;
  status?: string;
  progress?: number;
  stage?: string;
  step?: number;
  totalSteps?: number;
  error?: string;
}

interface UseDashboardStreamOptions {
  autoReconnect?: boolean;
  reconnectDelay?: number;
}

interface UseDashboardStreamReturn {
  isConnected: boolean;
  error: string | null;
}

export function useDashboardStream(
  onEvent: (event: DashboardEvent) => void,
  options: UseDashboardStreamOptions = {}
): UseDashboardStreamReturn {
  const { autoReconnect = true, reconnectDelay = 3000 } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const onEventRef = useRef(onEvent);

  // Keep onEvent ref updated
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const connect = useCallback(() => {
    disconnect();
    setError(null);

    const url = `${API_URL}/api/dashboard/stream`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    // Handle named events
    const handleEvent = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data) as DashboardEvent;
        onEventRef.current(data);
      } catch (e) {
        console.error("Failed to parse dashboard event:", e);
      }
    };

    eventSource.addEventListener("job_created", handleEvent);
    eventSource.addEventListener("job_updated", handleEvent);
    eventSource.addEventListener("job_progress", handleEvent);
    // ping events are just keepalives, no need to handle

    eventSource.onerror = () => {
      eventSource.close();
      setIsConnected(false);
      setError("Connection lost");

      if (autoReconnect) {
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectDelay);
      }
    };
  }, [autoReconnect, reconnectDelay, disconnect]);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    error,
  };
}
