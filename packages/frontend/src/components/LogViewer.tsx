"use client";

import { useEffect, useRef, useState } from "react";
import { Terminal, Download, Trash2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { LogEntry } from "@/types";
import { parseAnsi } from "@/lib/ansi";
import { useLogStream } from "@/lib/sse";

interface LogViewerProps {
  jobId: string;
}

export function LogViewer({ jobId }: LogViewerProps) {
  const { logs, isConnected, clearLogs } = useLogStream(jobId);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  const handleDownloadLogs = () => {
    const logText = logs
      .map(
        (log) =>
          `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] ${log.message}`
      )
      .join("\n");

    const blob = new Blob([logText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `job-${jobId}-logs.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearLogs = () => {
    clearLogs();
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Terminal className="w-5 h-5" />
              Live Logs
              {isConnected && (
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              )}
            </CardTitle>
            <CardDescription>Real-time job execution logs</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAutoScroll(!autoScroll)}
            >
              {autoScroll ? "Disable" : "Enable"} Auto-scroll
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownloadLogs}>
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
            <Button variant="outline" size="sm" onClick={handleClearLogs}>
              <Trash2 className="w-4 h-4 mr-2" />
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96 w-full rounded-md border bg-slate-950 p-4">
          <div className="font-mono text-sm space-y-1" ref={scrollRef}>
            {logs.length === 0 ? (
              <div className="text-gray-500 text-center py-8">
                Waiting for logs...
              </div>
            ) : (
              logs.map((log, index) => <LogLine key={index} log={log} />)
            )}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>
        <div className="mt-2 text-sm text-muted-foreground">
          {logs.length} log {logs.length === 1 ? "entry" : "entries"}
        </div>
      </CardContent>
    </Card>
  );
}

function LogLine({ log }: { log: LogEntry }) {
  const segments = parseAnsi(log.ansi || log.message);

  return (
    <div className="flex gap-2 hover:bg-slate-900/50 px-2 py-1 rounded">
      <span className="text-gray-500 text-xs flex-shrink-0 w-20">
        {log.timestamp.toLocaleTimeString()}
      </span>
      <span className="flex-1 break-all">
        {segments.map((segment, index) => (
          <span
            key={index}
            style={{
              color: segment.color || "#e2e8f0",
              fontWeight: segment.bold ? "bold" : "normal",
              fontStyle: segment.italic ? "italic" : "normal",
              textDecoration: segment.underline ? "underline" : "none",
            }}
          >
            {segment.text}
          </span>
        ))}
      </span>
    </div>
  );
}
