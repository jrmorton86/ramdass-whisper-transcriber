"use client";

import { useState } from "react";
import {
  FileAudio,
  Hash,
  Clock,
  Calendar,
  Download,
  Copy,
  Check,
  Edit2,
  Save,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Job } from "@/types";
import { toast } from "sonner";

interface JobDetailsProps {
  job: Job;
}

export function JobDetails({ job }: JobDetailsProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState(job.result?.text || "");
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(job.result?.text || "");
    setCopied(true);
    toast.success("Text copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSave = () => {
    // In a real app, this would call an API to save the edited text
    toast.success("Changes saved successfully");
    setIsEditing(false);
  };

  const handleDownload = () => {
    const blob = new Blob([job.result?.text || ""], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${job.name.replace(/\.[^/.]+$/, "")}_transcription.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Transcription downloaded");
  };

  return (
    <div className="space-y-6">
      {/* Job Info */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <CardTitle className="flex items-center gap-2">
                {job.type === "file" ? (
                  <FileAudio className="w-5 h-5 text-blue-500" />
                ) : (
                  <Hash className="w-5 h-5 text-purple-500" />
                )}
                {job.name}
              </CardTitle>
              <CardDescription>Job ID: {job.id}</CardDescription>
            </div>
            <Badge className={getStatusColor(job.status)}>
              {job.status.toUpperCase()}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <InfoItem
              icon={<Calendar className="w-4 h-4" />}
              label="Created"
              value={job.createdAt.toLocaleString()}
            />
            {job.startedAt && (
              <InfoItem
                icon={<Clock className="w-4 h-4" />}
                label="Started"
                value={job.startedAt.toLocaleString()}
              />
            )}
            {job.completedAt && (
              <InfoItem
                icon={<Clock className="w-4 h-4" />}
                label="Completed"
                value={job.completedAt.toLocaleString()}
              />
            )}
            {job.duration && (
              <InfoItem
                icon={<Clock className="w-4 h-4" />}
                label="Duration"
                value={`${job.duration.toFixed(2)}s`}
              />
            )}
          </div>

          {job.error && (
            <div className="mt-4 p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-sm font-medium text-red-800 dark:text-red-200">
                Error
              </p>
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                {job.error}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Transcription Result */}
      {job.result && (
        <>
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Transcription Result</CardTitle>
                  <CardDescription>
                    {job.result.segments.length} segments •{" "}
                    {job.result.metadata.duration.toFixed(1)}s •{" "}
                    {job.result.metadata.language} • {job.result.metadata.model}
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleCopy}>
                    {copied ? (
                      <>
                        <Check className="w-4 h-4 mr-2" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4 mr-2" />
                        Copy
                      </>
                    )}
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleDownload}>
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  {!isEditing ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setIsEditing(true)}
                    >
                      <Edit2 className="w-4 h-4 mr-2" />
                      Edit
                    </Button>
                  ) : (
                    <Button variant="default" size="sm" onClick={handleSave}>
                      <Save className="w-4 h-4 mr-2" />
                      Save
                    </Button>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {isEditing ? (
                <Textarea
                  value={editedText}
                  onChange={(e) => setEditedText(e.target.value)}
                  className="min-h-64 font-mono text-sm"
                  placeholder="Edit transcription..."
                />
              ) : (
                <ScrollArea className="h-64 w-full rounded-md border p-4">
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {job.result.text}
                  </p>
                </ScrollArea>
              )}
            </CardContent>
          </Card>

          {/* Segments */}
          <Card>
            <CardHeader>
              <CardTitle>Transcript Segments</CardTitle>
              <CardDescription>
                Timestamped segments of the transcription
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {job.result.segments.map((segment, index) => (
                  <div
                    key={index}
                    className="flex gap-4 p-3 bg-muted/50 rounded-lg"
                  >
                    <div className="flex-shrink-0 text-sm font-mono text-muted-foreground w-24">
                      {formatTimestamp(segment.start)} -{" "}
                      {formatTimestamp(segment.end)}
                    </div>
                    <div className="flex-1 text-sm">{segment.text}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function InfoItem({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-2">
      <div className="mt-1 text-muted-foreground">{icon}</div>
      <div>
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-sm font-medium">{value}</p>
      </div>
    </div>
  );
}

function getStatusColor(status: string): string {
  const colors = {
    pending: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    processing: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    cancelled: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
  };
  return colors[status as keyof typeof colors] || "bg-gray-100 text-gray-800";
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}:${secs.padStart(4, "0")}`;
}
