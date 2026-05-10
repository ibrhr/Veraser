import type {
  InpaintingJobResponse,
  MaskArtifactManifest,
  MaskingJobResponse,
  ObjectPromptListResponse,
  ObjectPromptResponse,
  Prompt,
  SubmitObjectPromptsResponse,
  VideoSessionResponse
} from "./types";

const API_ROOT = import.meta.env.VITE_API_ROOT ?? "";

export function apiUrl(path: string): string {
  if (path.startsWith("http")) {
    return path;
  }
  const root = API_ROOT.endsWith("/") ? API_ROOT.slice(0, -1) : API_ROOT;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${root}${normalizedPath}`;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function uploadVideo(video: File): Promise<VideoSessionResponse> {
  const formData = new FormData();
  formData.set("video", video);
  const response = await fetch(apiUrl("/api/v1/video-sessions"), {
    method: "POST",
    body: formData
  });
  return parseResponse<VideoSessionResponse>(response);
}

export async function listObjects(sessionId: string): Promise<ObjectPromptListResponse> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/objects`));
  return parseResponse<ObjectPromptListResponse>(response);
}

export async function createObject(sessionId: string, prompts: Prompt[]): Promise<SubmitObjectPromptsResponse> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/objects`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ objects: [{ prompts }] })
  });
  return parseResponse<SubmitObjectPromptsResponse>(response);
}

export async function updateObject(
  sessionId: string,
  objectId: string,
  prompts: Prompt[]
): Promise<ObjectPromptResponse> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/objects/${objectId}`), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompts })
  });
  return parseResponse<ObjectPromptResponse>(response);
}

export async function deleteObject(sessionId: string, objectId: string): Promise<void> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/objects/${objectId}`), {
    method: "DELETE"
  });
  if (!response.ok) {
    await parseResponse<unknown>(response);
  }
}

export async function startMaskingJob(sessionId: string): Promise<MaskingJobResponse> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/masking-jobs`), {
    method: "POST"
  });
  return parseResponse<MaskingJobResponse>(response);
}

export async function getMaskingJob(sessionId: string, jobId: string): Promise<MaskingJobResponse> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/masking-jobs/${jobId}`));
  return parseResponse<MaskingJobResponse>(response);
}

export async function getMaskManifest(sessionId: string, jobId: string): Promise<MaskArtifactManifest> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/masking-jobs/${jobId}/masks/manifest`));
  return parseResponse<MaskArtifactManifest>(response);
}

export async function startInpaintingJob(sessionId: string, maskingJobId: string): Promise<InpaintingJobResponse> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/masking-jobs/${maskingJobId}/inpainting-job`), {
    method: "POST"
  });
  return parseResponse<InpaintingJobResponse>(response);
}

export async function getInpaintingJob(sessionId: string, maskingJobId: string): Promise<InpaintingJobResponse> {
  const response = await fetch(apiUrl(`/api/v1/video-sessions/${sessionId}/masking-jobs/${maskingJobId}/inpainting-job`));
  return parseResponse<InpaintingJobResponse>(response);
}
