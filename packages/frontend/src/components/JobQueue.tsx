"use client";

import { useState } from "react";
import {
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  FileAudio,
  Hash,
  Trash2,
  Eye,
  X,
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
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Job, JobStatus } from "@/types";
import { api } from "@/lib/api";
import { toast } from "sonner";

interface JobQueueProps {
  jobs: Job[];
  onJobSelect: (job: Job) => void;
  onJobsChange: () => void;
}

export function JobQueue({ jobs, onJobSelect, onJobsChange }: JobQueueProps) {
  const [deleteJobId, setDeleteJobId] = useState<string | null>(null);
  const [cancelJobId, setCancelJobId] = useState<string | null>(null);

  const handleDelete = async (id: string) => {
    try {
      await api.deleteJob(id);
      toast.success("Job deleted successfully");
      onJobsChange();
    } catch (error) {
      toast.error("Failed to delete job");
    } finally {
      setDeleteJobId(null);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await api.cancelJob(id);
      toast.success("Job cancelled successfully");
      onJobsChange();
    } catch (error) {
      toast.error("Failed to cancel job");
    } finally {
      setCancelJobId(null);
    }
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Job Queue</CardTitle>
          <CardDescription>Monitor and manage transcription jobs</CardDescription>
        </CardHeader>
        <CardContent>
          {jobs.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <FileAudio className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No jobs yet. Create your first transcription job to get started.</p>
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Job</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Progress</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {jobs.map((job) => (
                    <TableRow
                      key={job.id}
                      className="cursor-pointer hover:bg-muted/50"
                    >
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-2">
                          {job.type === "file" ? (
                            <FileAudio className="w-4 h-4 text-blue-500" />
                          ) : (
                            <Hash className="w-4 h-4 text-purple-500" />
                          )}
                          <span className="truncate max-w-xs">{job.name}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">
                          {job.type === "file" ? "File" : "UUID"}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={job.status} />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2 min-w-40">
                          <Progress value={job.progress} className="h-2 flex-1" />
                          <span className="text-sm text-muted-foreground whitespace-nowrap">
                            {job.currentStep && job.totalSteps
                              ? `Step ${job.currentStep}/${job.totalSteps}`
                              : job.currentStage || `${job.progress}%`}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {formatTimeAgo(job.createdAt)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => onJobSelect(job)}
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                          {(job.status === "pending" ||
                            job.status === "processing") && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setCancelJobId(job.id)}
                            >
                              <X className="w-4 h-4" />
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setDeleteJobId(job.id)}
                          >
                            <Trash2 className="w-4 h-4 text-red-500" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteJobId} onOpenChange={() => setDeleteJobId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Job</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this job? This action cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteJobId && handleDelete(deleteJobId)}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Cancel Confirmation */}
      <AlertDialog open={!!cancelJobId} onOpenChange={() => setCancelJobId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel Job</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to cancel this job? The job will be stopped
              immediately.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>No</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => cancelJobId && handleCancel(cancelJobId)}
            >
              Yes, Cancel Job
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

function StatusBadge({ status }: { status: JobStatus }) {
  const configs = {
    pending: {
      icon: Clock,
      color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
      label: "Pending",
    },
    processing: {
      icon: Loader2,
      color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
      label: "Processing",
    },
    completed: {
      icon: CheckCircle2,
      color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      label: "Completed",
    },
    failed: {
      icon: XCircle,
      color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
      label: "Failed",
    },
    cancelled: {
      icon: XCircle,
      color: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
      label: "Cancelled",
    },
  };

  const config = configs[status];
  const Icon = config.icon;

  return (
    <Badge className={`${config.color} border-0`}>
      <Icon
        className={`w-3 h-3 mr-1 ${status === "processing" ? "animate-spin" : ""}`}
      />
      {config.label}
    </Badge>
  );
}

function formatTimeAgo(date: Date): string {
  const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);

  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
