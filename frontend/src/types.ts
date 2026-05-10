export type PointPrompt = {
  type: "point";
  x: number;
  y: number;
  label: "foreground" | "background";
};

export type BoxPrompt = {
  type: "box";
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type Prompt = PointPrompt | BoxPrompt;

export type StoredObjectPrompt = {
  object_id: string;
  client_object_id?: string | null;
  prompts: Prompt[];
};

export type OperationSpeedMetric = {
  name: string;
  label: string;
  elapsed_seconds: number;
  frames_processed?: number | null;
  fps?: number | null;
  seconds_per_frame?: number | null;
};

export type VideoSessionResponse = {
  session_id: string;
  status: "ready_for_prompts";
  created_at: string;
  video: {
    filename: string;
    content_type?: string | null;
    size_bytes: number;
    width?: number | null;
    height?: number | null;
    frame_count?: number | null;
    fps?: number | null;
    duration_seconds?: number | null;
  };
  first_frame: {
    width: number;
    height: number;
    content_type: "image/png";
    url: string;
  };
  performance: OperationSpeedMetric[];
};

export type ObjectPromptListResponse = {
  session_id: string;
  objects: StoredObjectPrompt[];
};

export type ObjectPromptResponse = {
  session_id: string;
  object: StoredObjectPrompt;
  preview_mask_url: string;
};

export type SubmitObjectPromptsResponse = {
  session_id: string;
  objects: StoredObjectPrompt[];
  model_status: "ready_for_masking";
};

export type MaskingJobResponse = {
  job_id: string;
  session_id: string;
  status: "pending" | "running" | "succeeded" | "failed";
  created_at: string;
  updated_at: string;
  frames_total?: number | null;
  frames_done: number;
  current_stage: string;
  error?: string | null;
  performance: OperationSpeedMetric[];
  manifest_url: string;
  processed_video_url: string;
};

export type InpaintingJobResponse = {
  job_id: string;
  session_id: string;
  masking_job_id: string;
  status: "pending" | "running" | "succeeded" | "failed";
  created_at: string;
  updated_at: string;
  frames_total?: number | null;
  frames_done: number;
  current_stage: string;
  error?: string | null;
  performance: OperationSpeedMetric[];
  processed_video_url: string;
};

export type MaskArtifactManifest = {
  session_id: string;
  job_id: string;
  frames_total: number;
  objects: string[];
  combined_masks_url: string;
  processed_video_url: string;
};
