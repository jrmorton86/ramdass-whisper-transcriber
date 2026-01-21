"use client";

import { useState } from "react";
import { Upload, Hash, Loader2, List } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { api } from "@/lib/api";
import { toast } from "sonner";

interface JobSubmissionProps {
  onJobCreated?: () => void;
}

export function JobSubmission({ onJobCreated }: JobSubmissionProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uuids, setUuids] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast.error("Please select a file");
      return;
    }

    setLoading(true);
    try {
      await api.uploadFile(file);
      toast.success("Job created successfully");
      setFile(null);
      onJobCreated?.();
    } catch (error) {
      toast.error("Failed to create job");
    } finally {
      setLoading(false);
    }
  };

  const handleUuidSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uuids.trim()) {
      toast.error("Please enter at least one UUID");
      return;
    }

    // Parse UUIDs - split by newline, comma, or semicolon
    const uuidList = uuids
      .split(/[\n,;]+/)
      .map((uuid) => uuid.trim())
      .filter((uuid) => uuid.length > 0);

    if (uuidList.length === 0) {
      toast.error("Please enter at least one valid UUID");
      return;
    }

    setLoading(true);
    try {
      // Create a job for each UUID
      const promises = uuidList.map((uuid) =>
        api.createJob({
          type: "uuid",
          input: uuid,
        })
      );

      await Promise.all(promises);

      if (uuidList.length === 1) {
        toast.success("Job created successfully");
      } else {
        toast.success(`${uuidList.length} jobs created successfully`);
      }

      setUuids("");
      onJobCreated?.();
    } catch (error) {
      toast.error("Failed to create jobs");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create Transcription Job</CardTitle>
        <CardDescription>
          Upload an audio file or provide an Intelligence Bank UUID
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="file" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="file">
              <Upload className="w-4 h-4 mr-2" />
              File Upload
            </TabsTrigger>
            <TabsTrigger value="uuid">
              <Hash className="w-4 h-4 mr-2" />
              UUID Entry
            </TabsTrigger>
          </TabsList>

          <TabsContent value="file" className="space-y-4">
            <form onSubmit={handleFileSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="file">Audio File</Label>
                <div
                  className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                    dragActive
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-950"
                      : "border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input
                    id="file"
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    accept="audio/*,video/*"
                    onChange={handleFileChange}
                  />
                  <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  {file ? (
                    <div>
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-muted-foreground">
                        Drag and drop your audio file here, or click to browse
                      </p>
                      <p className="text-sm text-muted-foreground mt-2">
                        Supports MP3, WAV, M4A, FLAC, and more
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <Button type="submit" className="w-full" disabled={!file || loading}>
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Creating Job...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Create Job
                  </>
                )}
              </Button>
            </form>
          </TabsContent>

          <TabsContent value="uuid" className="space-y-4">
            <form onSubmit={handleUuidSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="uuid">Intelligence Bank UUIDs</Label>
                <Textarea
                  id="uuid"
                  placeholder={`Enter UUIDs (one per line or comma-separated)\nabc-123-def-456\nxyz-789-ghi-012\n...`}
                  value={uuids}
                  onChange={(e) => setUuids(e.target.value)}
                  className="font-mono min-h-32"
                  rows={8}
                />
                <p className="text-sm text-muted-foreground">
                  Enter one or more UUIDs from Intelligence Bank. Separate
                  multiple UUIDs with newlines, commas, or semicolons.
                </p>
                {uuids.trim() && (
                  <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                    {uuids.split(/[\n,;]+/).filter((u) => u.trim()).length}{" "}
                    UUID(s) detected
                  </p>
                )}
              </div>

              <Button
                type="submit"
                className="w-full"
                disabled={!uuids.trim() || loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Creating Jobs...
                  </>
                ) : (
                  <>
                    <List className="w-4 h-4 mr-2" />
                    Create Batch Jobs
                  </>
                )}
              </Button>
            </form>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
