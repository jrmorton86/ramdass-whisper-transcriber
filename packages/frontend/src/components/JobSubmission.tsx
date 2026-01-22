"use client";

import { useState, useRef, useCallback } from "react";
import {
  Upload,
  Hash,
  Loader2,
  List,
  ChevronDown,
  ChevronRight,
  Check,
  X,
  Circle,
  Folder,
  Trash2,
  FolderOpen,
} from "lucide-react";
import { Button } from "@/components/ui/button";
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
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { toast } from "sonner";
import {
  UploadFile,
  generateFileId,
  uploadFilesWithProgress,
  extractFilesFromDataTransfer,
  createUploadFiles,
  calculateUploadProgress,
} from "@/lib/upload";

interface JobSubmissionProps {
  onJobCreated?: () => void;
}

// Concurrency limit for uploads
const UPLOAD_CONCURRENCY = 3;

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

export function JobSubmission({ onJobCreated }: JobSubmissionProps) {
  // UUID tab state
  const [uuids, setUuids] = useState("");
  const [uuidLoading, setUuidLoading] = useState(false);

  // File upload state
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [folderMode, setFolderMode] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [filesExpanded, setFilesExpanded] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Calculate overall progress
  const progress = calculateUploadProgress(files);
  const overallPercent =
    progress.totalBytes > 0
      ? Math.round((progress.uploadedBytes / progress.totalBytes) * 100)
      : 0;

  // Add files to the queue
  const addFiles = useCallback((newFiles: File[]) => {
    const uploadFiles = createUploadFiles(newFiles);
    setFiles((prev) => [...prev, ...uploadFiles]);
  }, []);

  // Remove a file from the queue
  const removeFile = useCallback((fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  }, []);

  // Clear all files
  const clearAllFiles = useCallback(() => {
    setFiles([]);
  }, []);

  // Handle file input change
  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        const fileArray = Array.from(e.target.files);
        addFiles(fileArray);
      }
      // Reset input so the same files can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [addFiles]
  );

  // Handle drag events
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  // Handle drop
  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      const extractedFiles = await extractFilesFromDataTransfer(e.dataTransfer);
      if (extractedFiles.length > 0) {
        addFiles(extractedFiles);
      }
    },
    [addFiles]
  );

  // Handle upload
  const handleUpload = async () => {
    const pendingFiles = files.filter((f) => f.status === "pending");
    if (pendingFiles.length === 0) {
      toast.error("No files to upload");
      return;
    }

    setIsUploading(true);

    // Create upload files with "uploading" status for the upload function
    // Note: We pass pendingFiles directly since state updates are async
    const filesToUpload: UploadFile[] = pendingFiles.map((f) => ({
      ...f,
      status: "uploading" as const,
    }));

    // Update UI to show uploading status
    setFiles((prev) =>
      prev.map((f) =>
        f.status === "pending" ? { ...f, status: "uploading" as const } : f
      )
    );

    // Generate a batch ID for this upload session
    const batchId = generateFileId();

    let completedCount = 0;
    let failedCount = 0;

    try {
      await uploadFilesWithProgress(
        filesToUpload,
        batchId,
        UPLOAD_CONCURRENCY,
        // onFileProgress
        (fileId, progressValue) => {
          setFiles((prev) =>
            prev.map((f) =>
              f.id === fileId ? { ...f, progress: progressValue } : f
            )
          );
        },
        // onFileComplete
        (fileId, jobId) => {
          completedCount++;
          setFiles((prev) =>
            prev.map((f) =>
              f.id === fileId
                ? { ...f, status: "complete" as const, progress: 100, jobId }
                : f
            )
          );
        },
        // onFileError
        (fileId, error) => {
          failedCount++;
          setFiles((prev) =>
            prev.map((f) =>
              f.id === fileId ? { ...f, status: "error" as const, error } : f
            )
          );
        }
      );

      // Show toast based on results
      if (failedCount === 0) {
        toast.success(
          `Successfully uploaded ${completedCount} file${completedCount > 1 ? "s" : ""}`
        );
      } else if (completedCount > 0) {
        toast.warning(`Uploaded ${completedCount} files, ${failedCount} failed`);
      } else {
        toast.error("All uploads failed");
      }

      onJobCreated?.();
    } catch (error) {
      toast.error("Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  // Handle UUID submit (preserved from original)
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

    setUuidLoading(true);
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
      setUuidLoading(false);
    }
  };

  // Get status icon for a file
  const getStatusIcon = (file: UploadFile) => {
    switch (file.status) {
      case "pending":
        return <Circle className="w-4 h-4 text-gray-400" />;
      case "uploading":
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case "complete":
        return <Check className="w-4 h-4 text-green-500" />;
      case "error":
        return <X className="w-4 h-4 text-red-500" />;
    }
  };

  // Get status text for a file
  const getStatusText = (file: UploadFile) => {
    switch (file.status) {
      case "pending":
        return "Pending";
      case "uploading":
        return `Uploading... ${file.progress}%`;
      case "complete":
        return "Completed";
      case "error":
        return file.error || "Failed";
    }
  };

  // Check if we can remove files (only before upload starts)
  const canModifyFiles = !isUploading;
  const hasFiles = files.length > 0;
  const hasPendingFiles = files.some((f) => f.status === "pending");
  const totalSize = files.reduce((sum, f) => sum + f.file.size, 0);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create Transcription Job</CardTitle>
        <CardDescription>
          Upload audio files or provide Intelligence Bank UUIDs
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
            {/* Folder mode toggle */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="folder-mode"
                checked={folderMode}
                onChange={(e) => setFolderMode(e.target.checked)}
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                disabled={isUploading}
              />
              <Label
                htmlFor="folder-mode"
                className="flex items-center gap-2 cursor-pointer"
              >
                <FolderOpen className="w-4 h-4" />
                Enable folder mode
              </Label>
            </div>

            {/* Drop zone */}
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
                ref={fileInputRef}
                type="file"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                accept="audio/*,video/*"
                multiple
                onChange={handleFileChange}
                disabled={isUploading}
                {...(folderMode
                  ? {
                      webkitdirectory: "true",
                      directory: "",
                    }
                  : {})}
              />
              {folderMode ? (
                <Folder className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              ) : (
                <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              )}
              {hasFiles ? (
                <div>
                  <p className="font-medium">
                    {files.length} file{files.length > 1 ? "s" : ""} selected
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {formatBytes(totalSize)}
                  </p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Drop more files to add them
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-muted-foreground">
                    {folderMode
                      ? "Drag folders here or click to browse"
                      : "Drag files/folders here or click to browse"}
                  </p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Supports MP3, WAV, M4A, FLAC, and more
                  </p>
                </div>
              )}
            </div>

            {/* Overall progress bar - only show during/after upload */}
            {hasFiles && (isUploading || progress.completedFiles > 0) && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Overall Progress:</span>
                  <span>
                    {overallPercent}% ({formatBytes(progress.uploadedBytes)} /{" "}
                    {formatBytes(progress.totalBytes)})
                  </span>
                </div>
                <Progress value={overallPercent} className="h-2" />
              </div>
            )}

            {/* File list */}
            {hasFiles && (
              <div className="border rounded-lg">
                {/* Collapsible header */}
                <button
                  type="button"
                  className="w-full flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  onClick={() => setFilesExpanded(!filesExpanded)}
                >
                  <span className="flex items-center gap-2 font-medium">
                    {filesExpanded ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                    Files ({files.length})
                  </span>
                  {progress.completedFiles > 0 && (
                    <span className="text-sm text-muted-foreground">
                      {progress.completedFiles} completed
                      {progress.failedFiles > 0 &&
                        `, ${progress.failedFiles} failed`}
                    </span>
                  )}
                </button>

                {/* File list content */}
                {filesExpanded && (
                  <div className="border-t max-h-64 overflow-y-auto">
                    {files.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between p-3 border-b last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-800"
                      >
                        <div className="flex items-center gap-3 min-w-0 flex-1">
                          {getStatusIcon(file)}
                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-medium truncate">
                              {file.file.name}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {formatBytes(file.file.size)} -{" "}
                              {getStatusText(file)}
                            </p>
                          </div>
                        </div>
                        {canModifyFiles && file.status === "pending" && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => removeFile(file.id)}
                            className="ml-2 text-gray-400 hover:text-red-500"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Action buttons */}
            <div className="flex justify-between items-center">
              <Button
                type="button"
                variant="outline"
                onClick={clearAllFiles}
                disabled={!hasFiles || isUploading}
              >
                Clear All
              </Button>
              <Button
                type="button"
                onClick={handleUpload}
                disabled={!hasPendingFiles || isUploading}
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload Files
                  </>
                )}
              </Button>
            </div>
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
                disabled={!uuids.trim() || uuidLoading}
              >
                {uuidLoading ? (
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
