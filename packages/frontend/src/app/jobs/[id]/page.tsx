"use client";

import { useState, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { JobDetails } from "@/components/JobDetails";
import { LogViewer } from "@/components/LogViewer";
import { api } from "@/lib/api";
import { Job } from "@/types";

interface JobPageProps {
  params: Promise<{ id: string }>;
}

export default function JobPage({ params }: JobPageProps) {
  const { id } = use(params);
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    fetchJob();
    // Poll for updates if job is processing
    const interval = setInterval(() => {
      if (job?.status === "processing" || job?.status === "pending") {
        fetchJob();
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [id, job?.status]);

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

        {/* Log Viewer - show for processing/pending jobs */}
        {(job.status === "processing" || job.status === "pending") && (
          <LogViewer jobId={job.id} />
        )}
      </div>
    </div>
  );
}
