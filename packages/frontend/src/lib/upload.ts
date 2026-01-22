/**
 * Multi-file upload utilities with progress tracking
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Supported audio/video MIME types for filtering
const SUPPORTED_MEDIA_TYPES = [
  "audio/",
  "video/",
  "audio/mpeg",
  "audio/mp3",
  "audio/wav",
  "audio/ogg",
  "audio/flac",
  "audio/aac",
  "audio/m4a",
  "audio/webm",
  "video/mp4",
  "video/webm",
  "video/ogg",
  "video/quicktime",
  "video/x-msvideo",
  "video/x-matroska",
];

// Supported file extensions (fallback when MIME type is not available)
const SUPPORTED_EXTENSIONS = [
  ".mp3",
  ".wav",
  ".ogg",
  ".flac",
  ".aac",
  ".m4a",
  ".wma",
  ".mp4",
  ".webm",
  ".mkv",
  ".avi",
  ".mov",
  ".wmv",
  ".m4v",
];

// ============================================================================
// Types
// ============================================================================

export interface UploadFile {
  id: string; // Unique ID for tracking
  file: File; // The actual File object
  status: "pending" | "uploading" | "complete" | "error";
  progress: number; // 0-100
  jobId?: string; // Job ID after successful upload
  error?: string; // Error message if failed
}

export interface UploadProgress {
  totalFiles: number;
  completedFiles: number;
  failedFiles: number;
  totalBytes: number;
  uploadedBytes: number;
  currentFile?: string;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate a unique file ID for tracking uploads
 */
export function generateFileId(): string {
  // Use crypto.randomUUID if available (modern browsers), otherwise fallback
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback: timestamp + random string
  return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * Check if a file is a supported audio/video file
 */
function isSupportedMediaFile(file: File): boolean {
  // Check MIME type first
  if (file.type) {
    const isSupported = SUPPORTED_MEDIA_TYPES.some(
      (type) => file.type.startsWith(type) || file.type === type
    );
    if (isSupported) return true;
  }

  // Fallback to extension check
  const fileName = file.name.toLowerCase();
  return SUPPORTED_EXTENSIONS.some((ext) => fileName.endsWith(ext));
}

// ============================================================================
// Upload Functions
// ============================================================================

/**
 * Upload a single file with progress tracking using XHR
 */
export async function uploadFileWithProgress(
  file: File,
  batchId: string | null,
  onProgress: (progress: number) => void
): Promise<{ success: boolean; jobId?: string; error?: string }> {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    // Build URL with optional batch_id query parameter
    let url = `${API_URL}/api/upload`;
    if (batchId) {
      url += `?batch_id=${encodeURIComponent(batchId)}`;
    }

    // Track upload progress
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        const progress = Math.round((event.loaded / event.total) * 100);
        onProgress(progress);
      }
    });

    // Handle completion
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve({
            success: true,
            jobId: response.id || response.job_id,
          });
        } catch {
          resolve({
            success: false,
            error: "Invalid response from server",
          });
        }
      } else {
        let errorMessage = `Upload failed (HTTP ${xhr.status})`;
        try {
          const errorResponse = JSON.parse(xhr.responseText);
          if (errorResponse.detail) {
            errorMessage = errorResponse.detail;
          }
        } catch {
          // Keep default error message
        }
        resolve({
          success: false,
          error: errorMessage,
        });
      }
    });

    // Handle network errors
    xhr.addEventListener("error", () => {
      resolve({
        success: false,
        error: "Network error - please check your connection",
      });
    });

    // Handle timeout
    xhr.addEventListener("timeout", () => {
      resolve({
        success: false,
        error: "Upload timed out",
      });
    });

    // Handle abort
    xhr.addEventListener("abort", () => {
      resolve({
        success: false,
        error: "Upload cancelled",
      });
    });

    // Send the request
    xhr.open("POST", url);
    xhr.send(formData);
  });
}

/**
 * Upload multiple files with configurable concurrency and progress tracking
 * Uses a Promise pool pattern to limit concurrent uploads
 */
export async function uploadFilesWithProgress(
  files: UploadFile[],
  batchId: string | null,
  concurrency: number,
  onFileProgress: (fileId: string, progress: number) => void,
  onFileComplete: (fileId: string, jobId: string) => void,
  onFileError: (fileId: string, error: string) => void
): Promise<void> {
  // Filter to only pending or uploading files (caller may have already set status)
  const filesToUpload = files.filter(
    (f) => f.status === "pending" || f.status === "uploading"
  );

  if (filesToUpload.length === 0) {
    return;
  }

  // Create a queue of files to upload
  const queue = [...filesToUpload];
  const inProgress = new Set<string>();

  // Process files with concurrency limit
  async function processNext(): Promise<void> {
    if (queue.length === 0) {
      return;
    }

    const uploadFile = queue.shift();
    if (!uploadFile) {
      return;
    }

    inProgress.add(uploadFile.id);

    try {
      const result = await uploadFileWithProgress(
        uploadFile.file,
        batchId,
        (progress) => onFileProgress(uploadFile.id, progress)
      );

      if (result.success && result.jobId) {
        onFileComplete(uploadFile.id, result.jobId);
      } else {
        onFileError(uploadFile.id, result.error || "Upload failed");
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Unknown error occurred";
      onFileError(uploadFile.id, errorMessage);
    } finally {
      inProgress.delete(uploadFile.id);
    }

    // Process next file in queue
    await processNext();
  }

  // Start concurrent uploads up to the limit
  const workers = Array(Math.min(concurrency, pendingFiles.length))
    .fill(null)
    .map(() => processNext());

  // Wait for all workers to complete
  await Promise.all(workers);
}

// ============================================================================
// Drag and Drop File Extraction
// ============================================================================

/**
 * FileSystemEntry interface for webkit drag and drop API
 */
interface FileSystemEntry {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  file?: (callback: (file: File) => void, errorCallback?: (err: Error) => void) => void;
  createReader?: () => FileSystemDirectoryReader;
}

interface FileSystemDirectoryReader {
  readEntries: (
    callback: (entries: FileSystemEntry[]) => void,
    errorCallback?: (err: Error) => void
  ) => void;
}

/**
 * Read all entries from a directory (handles batching)
 */
async function readAllDirectoryEntries(
  reader: FileSystemDirectoryReader
): Promise<FileSystemEntry[]> {
  const entries: FileSystemEntry[] = [];

  // readEntries returns batches, need to call until empty
  const readBatch = (): Promise<FileSystemEntry[]> => {
    return new Promise((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
  };

  let batch: FileSystemEntry[];
  do {
    batch = await readBatch();
    entries.push(...batch);
  } while (batch.length > 0);

  return entries;
}

/**
 * Recursively extract files from a FileSystemEntry
 */
async function extractFilesFromEntry(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile && entry.file) {
    return new Promise((resolve) => {
      entry.file!((file) => {
        resolve([file]);
      }, () => {
        resolve([]);
      });
    });
  }

  if (entry.isDirectory && entry.createReader) {
    const reader = entry.createReader();
    const entries = await readAllDirectoryEntries(reader);

    const nestedFiles = await Promise.all(
      entries.map((e) => extractFilesFromEntry(e))
    );

    return nestedFiles.flat();
  }

  return [];
}

/**
 * Extract files from a DataTransfer object (drag and drop)
 * Handles folder drops via webkitGetAsEntry and filters to audio/video files
 */
export async function extractFilesFromDataTransfer(
  dataTransfer: DataTransfer
): Promise<File[]> {
  const files: File[] = [];

  // Check if webkitGetAsEntry is available (for folder support)
  const items = dataTransfer.items;
  const hasWebkitEntry =
    items.length > 0 &&
    items[0].webkitGetAsEntry !== undefined;

  if (hasWebkitEntry) {
    // Use webkitGetAsEntry for folder support
    const entries: FileSystemEntry[] = [];

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === "file") {
        const entry = item.webkitGetAsEntry() as FileSystemEntry | null;
        if (entry) {
          entries.push(entry);
        }
      }
    }

    // Extract files from all entries (including nested folders)
    const extractedFiles = await Promise.all(
      entries.map((entry) => extractFilesFromEntry(entry))
    );

    files.push(...extractedFiles.flat());
  } else {
    // Fallback: use dataTransfer.files (no folder support)
    for (let i = 0; i < dataTransfer.files.length; i++) {
      files.push(dataTransfer.files[i]);
    }
  }

  // Filter to only supported audio/video files
  return files.filter(isSupportedMediaFile);
}

/**
 * Create UploadFile objects from a list of Files
 */
export function createUploadFiles(files: File[]): UploadFile[] {
  return files.map((file) => ({
    id: generateFileId(),
    file,
    status: "pending" as const,
    progress: 0,
  }));
}

/**
 * Calculate overall upload progress from a list of upload files
 */
export function calculateUploadProgress(files: UploadFile[]): UploadProgress {
  const totalFiles = files.length;
  const completedFiles = files.filter((f) => f.status === "complete").length;
  const failedFiles = files.filter((f) => f.status === "error").length;

  const totalBytes = files.reduce((sum, f) => sum + f.file.size, 0);
  const uploadedBytes = files.reduce((sum, f) => {
    if (f.status === "complete") {
      return sum + f.file.size;
    }
    if (f.status === "uploading") {
      return sum + (f.file.size * f.progress) / 100;
    }
    return sum;
  }, 0);

  const currentFile = files.find((f) => f.status === "uploading")?.file.name;

  return {
    totalFiles,
    completedFiles,
    failedFiles,
    totalBytes,
    uploadedBytes,
    currentFile,
  };
}
