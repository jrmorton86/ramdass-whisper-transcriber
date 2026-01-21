"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { LogEntry } from "@/types";
import { getLogsStreamUrl } from "./api";

interface UseLogStreamOptions {
  maxLogs?: number;
  autoReconnect?: boolean;
  reconnectDelay?: number;
}

interface UseLogStreamReturn {
  logs: LogEntry[];
  isConnected: boolean;
  error: string | null;
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

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

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

  const connect = useCallback(() => {
    if (!jobId) return;

    disconnect();
    setError(null);

    const url = getLogsStreamUrl(jobId);
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "log") {
          const logEntry: LogEntry = {
            timestamp: new Date(data.timestamp),
            level: data.level || "info",
            message: data.message,
            ansi: data.ansi || data.message,
          };

          setLogs((prev) => {
            const newLogs = [...prev, logEntry];
            // Trim to maxLogs
            if (newLogs.length > maxLogs) {
              return newLogs.slice(-maxLogs);
            }
            return newLogs;
          });
        } else if (data.type === "complete" || data.type === "error") {
          // Job finished, close connection
          eventSource.close();
          setIsConnected(false);
        }
      } catch (e) {
        console.error("Failed to parse SSE message:", e);
      }
    };

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
  }, [jobId, maxLogs, autoReconnect, reconnectDelay, disconnect]);

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
    connect,
    disconnect,
    clearLogs,
  };
}
