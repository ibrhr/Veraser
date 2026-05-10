import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KonvaImage, Layer, Rect, Stage } from "react-konva";
import type Konva from "konva";
import {
  BoxSelect,
  Check,
  CircleDot,
  Eraser,
  Eye,
  Loader2,
  MousePointer2,
  Play,
  Trash2,
  Upload,
  X
} from "lucide-react";
import {
  apiUrl,
  createObject,
  deleteObject,
  getInpaintingJob,
  getMaskManifest,
  getMaskingJob,
  listObjects,
  startInpaintingJob,
  startMaskingJob,
  updateObject,
  uploadVideo
} from "./api";
import type {
  BoxPrompt,
  InpaintingJobResponse,
  MaskArtifactManifest,
  MaskingJobResponse,
  OperationSpeedMetric,
  PointPrompt,
  Prompt,
  StoredObjectPrompt,
  VideoSessionResponse
} from "./types";

type Tool = "point" | "box" | "select";
type PointLabel = "foreground" | "background";

type DraftBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type ImageLoadState = {
  image: HTMLImageElement | null;
  error: string | null;
};

function useHtmlImage(src: string | null, label: string): ImageLoadState {
  const [state, setState] = useState<ImageLoadState>({ image: null, error: null });

  useEffect(() => {
    if (!src) {
      setState({ image: null, error: null });
      return;
    }
    const imageSrc = src;
    let cancelled = false;
    let objectUrl: string | null = null;
    const controller = new AbortController();
    const nextImage = new window.Image();
    nextImage.crossOrigin = "anonymous";
    setState({ image: null, error: null });

    async function loadImage() {
      try {
        const response = await fetch(imageSrc, { cache: "no-store", signal: controller.signal });
        if (!response.ok) {
          throw new Error(await responseErrorMessage(response, label));
        }
        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        nextImage.onload = () => {
          if (!cancelled) {
            setState({ image: nextImage, error: null });
          }
        };
        nextImage.onerror = () => {
          if (!cancelled) {
            setState({ image: null, error: `Unable to decode ${label}.` });
          }
        };
        nextImage.src = objectUrl;
      } catch (cause) {
        if (!cancelled && cause instanceof Error && cause.name !== "AbortError") {
          setState({ image: null, error: cause.message });
        }
      }
    }

    void loadImage();
    return () => {
      cancelled = true;
      controller.abort();
      nextImage.onload = null;
      nextImage.onerror = null;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [label, src]);

  return state;
}

async function responseErrorMessage(response: Response, label: string) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") {
      return body.detail;
    }
    return JSON.stringify(body.detail ?? body);
  } catch {
    const text = await response.text();
    return text || `Unable to load ${label}: ${response.status} ${response.statusText}`;
  }
}

function clamp(value: number, max: number) {
  return Math.min(Math.max(Math.round(value), 0), max);
}

function normalizeBox(box: DraftBox, width: number, height: number): BoxPrompt | null {
  const x1 = clamp(Math.min(box.x1, box.x2), width);
  const y1 = clamp(Math.min(box.y1, box.y2), height);
  const x2 = clamp(Math.max(box.x1, box.x2), width);
  const y2 = clamp(Math.max(box.y1, box.y2), height);
  if (x2 - x1 < 4 || y2 - y1 < 4) {
    return null;
  }
  return { type: "box", x1, y1, x2, y2 };
}

function promptKey(prompt: Prompt, index: number) {
  if (prompt.type === "point") {
    return `point-${prompt.x}-${prompt.y}-${prompt.label}-${index}`;
  }
  return `box-${prompt.x1}-${prompt.y1}-${prompt.x2}-${prompt.y2}-${index}`;
}

function promptSummary(prompt: Prompt) {
  if (prompt.type === "point") {
    return `${prompt.label} point (${prompt.x}, ${prompt.y})`;
  }
  return `box (${prompt.x1}, ${prompt.y1}) -> (${prompt.x2}, ${prompt.y2})`;
}

function formatSeconds(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return `${value.toFixed(value >= 10 ? 2 : 3)}s`;
}

function formatRate(value: number | null | undefined, suffix: string) {
  if (value === null || value === undefined) {
    return "-";
  }
  return `${value.toFixed(value >= 10 ? 2 : 3)} ${suffix}`;
}

function SpeedMetrics({ title, metrics }: { title: string; metrics: OperationSpeedMetric[] }) {
  if (metrics.length === 0) {
    return null;
  }
  return (
    <div className="speed-panel">
      <h2>{title}</h2>
      <div className="speed-list">
        {metrics.map((metric) => (
          <div className="speed-row" key={metric.name}>
            <strong>{metric.label}</strong>
            <span>{formatSeconds(metric.elapsed_seconds)}</span>
            <span>{formatRate(metric.fps, "FPS")}</span>
            <span>{formatRate(metric.seconds_per_frame, "s/frame")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState<VideoSessionResponse | null>(null);
  const [objects, setObjects] = useState<StoredObjectPrompt[]>([]);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [draftPrompts, setDraftPrompts] = useState<Prompt[]>([]);
  const [tool, setTool] = useState<Tool>("box");
  const [pointLabel, setPointLabel] = useState<PointLabel>("foreground");
  const [previewObjectId, setPreviewObjectId] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);
  const [job, setJob] = useState<MaskingJobResponse | null>(null);
  const [inpaintingJob, setInpaintingJob] = useState<InpaintingJobResponse | null>(null);
  const [manifest, setManifest] = useState<MaskArtifactManifest | null>(null);
  const [maskFrameIndex, setMaskFrameIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [canvasWidth, setCanvasWidth] = useState(900);
  const [draftBox, setDraftBox] = useState<DraftBox | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canvasHostRef = useRef<HTMLDivElement | null>(null);

  const firstFrameUrl = session ? apiUrl(session.first_frame.url) : null;
  const sourceVideoUrl = session ? apiUrl(`/api/v1/video-sessions/${session.session_id}/source-video`) : null;
  const processedVideoUrl =
    inpaintingJob?.status === "succeeded" ? apiUrl(inpaintingJob.processed_video_url) : null;
  const previewUrl = session && previewObjectId ? apiUrl(`/api/v1/video-sessions/${session.session_id}/objects/${previewObjectId}/preview-mask`) : null;
  const trackedMaskUrl =
    session && job?.status === "succeeded"
      ? apiUrl(`/api/v1/video-sessions/${session.session_id}/masking-jobs/${job.job_id}/masks/combined/${maskFrameIndex}`)
      : null;

  const firstFrameImageState = useHtmlImage(firstFrameUrl, "first frame");
  const previewImageState = useHtmlImage(previewUrl ? `${previewUrl}?t=${previewNonce}` : null, "preview mask");
  const trackedMaskImageState = useHtmlImage(trackedMaskUrl, "tracked mask");
  const firstFrameImage = firstFrameImageState.image;
  const previewImage = previewImageState.image;
  const trackedMaskImage = trackedMaskImageState.image;

  const frameSize = useMemo(() => {
    if (!session) {
      return { width: 1280, height: 720 };
    }
    return { width: session.first_frame.width, height: session.first_frame.height };
  }, [session]);

  const scale = Math.min(canvasWidth / frameSize.width, 1);
  const stageWidth = Math.max(320, Math.round(frameSize.width * scale));
  const stageHeight = Math.max(180, Math.round(frameSize.height * scale));

  const selectedObject = objects.find((object) => object.object_id === selectedObjectId) ?? null;
  const visiblePrompts = previewImage ? [] : draftPrompts;

  const refreshObjects = useCallback(async (sessionId: string) => {
    const response = await listObjects(sessionId);
    setObjects(response.objects);
  }, []);

  useEffect(() => {
    const element = canvasHostRef.current;
    if (!element) {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      setCanvasWidth(Math.max(320, Math.floor(entry.contentRect.width)));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const imageError = firstFrameImageState.error ?? previewImageState.error ?? trackedMaskImageState.error;
    if (imageError) {
      setError(imageError);
    }
  }, [firstFrameImageState.error, previewImageState.error, trackedMaskImageState.error]);

  useEffect(() => {
    if (!session || !job || job.status === "succeeded" || job.status === "failed") {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await getMaskingJob(session.session_id, job.job_id);
        setJob(nextJob);
        if (nextJob.status === "succeeded") {
          const nextManifest = await getMaskManifest(session.session_id, nextJob.job_id);
          setManifest(nextManifest);
        }
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Unable to poll masking job.");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job, session]);

  useEffect(() => {
    if (!session || !job || !inpaintingJob || inpaintingJob.status === "succeeded" || inpaintingJob.status === "failed") {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await getInpaintingJob(session.session_id, job.job_id);
        setInpaintingJob(nextJob);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Unable to poll inpainting job.");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [inpaintingJob, job, session]);

  function stagePoint(stage: Konva.Stage): { x: number; y: number } | null {
    const pointer = stage.getPointerPosition();
    if (!pointer) {
      return null;
    }
    return {
      x: clamp(pointer.x / scale, frameSize.width),
      y: clamp(pointer.y / scale, frameSize.height)
    };
  }

  async function handleUpload(file: File | undefined) {
    if (!file) {
      return;
    }
    setBusy(true);
    setError(null);
    setJob(null);
    setInpaintingJob(null);
    setManifest(null);
    setObjects([]);
    setDraftPrompts([]);
    setSelectedObjectId(null);
    setPreviewObjectId(null);
    try {
      const nextSession = await uploadVideo(file);
      setSession(nextSession);
      await refreshObjects(nextSession.session_id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function saveDraft() {
    if (!session || draftPrompts.length === 0) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      if (selectedObjectId) {
        const response = await updateObject(session.session_id, selectedObjectId, draftPrompts);
        setPreviewObjectId(response.object.object_id);
        setDraftPrompts(response.object.prompts);
        setPreviewNonce((current) => current + 1);
      } else {
        const response = await createObject(session.session_id, draftPrompts);
        const createdObject = response.objects[response.objects.length - 1] ?? null;
        setSelectedObjectId(createdObject?.object_id ?? null);
        setPreviewObjectId(createdObject?.object_id ?? null);
        setDraftPrompts(createdObject?.prompts ?? []);
        setPreviewNonce((current) => current + 1);
      }
      await refreshObjects(session.session_id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to save object.");
    } finally {
      setBusy(false);
    }
  }

  async function removeObject(objectId: string) {
    if (!session) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await deleteObject(session.session_id, objectId);
      if (selectedObjectId === objectId) {
        setSelectedObjectId(null);
        setDraftPrompts([]);
      }
      if (previewObjectId === objectId) {
        setPreviewObjectId(null);
      }
      await refreshObjects(session.session_id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to delete object.");
    } finally {
      setBusy(false);
    }
  }

  async function runMasking() {
    if (!session) {
      return;
    }
    setBusy(true);
    setError(null);
    setInpaintingJob(null);
    setManifest(null);
    try {
      const nextJob = await startMaskingJob(session.session_id);
      setJob(nextJob);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to start masking job.");
    } finally {
      setBusy(false);
    }
  }

  async function runInpainting() {
    if (!session || !job || job.status !== "succeeded") {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const nextJob = await startInpaintingJob(session.session_id, job.job_id);
      setInpaintingJob(nextJob);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to start inpainting job.");
    } finally {
      setBusy(false);
    }
  }

  function selectObject(object: StoredObjectPrompt) {
    setError(null);
    setSelectedObjectId(object.object_id);
    setDraftPrompts(object.prompts);
    setPreviewObjectId(null);
  }

  function previewObject(object: StoredObjectPrompt) {
    setError(null);
    setSelectedObjectId(object.object_id);
    setDraftPrompts(object.prompts);
    setPreviewObjectId(object.object_id);
    setPreviewNonce((current) => current + 1);
  }

  function startNewObject() {
    setError(null);
    setSelectedObjectId(null);
    setDraftPrompts([]);
    setPreviewObjectId(null);
  }

  function handleStagePointerDown(event: Konva.KonvaEventObject<PointerEvent>) {
    if (!session || tool === "select") {
      return;
    }
    const stage = event.target.getStage();
    if (!stage) {
      return;
    }
    const point = stagePoint(stage);
    if (!point) {
      return;
    }
    if (tool === "point") {
      setPreviewObjectId(null);
      const prompt: PointPrompt = {
        type: "point",
        x: clamp(point.x, frameSize.width - 1),
        y: clamp(point.y, frameSize.height - 1),
        label: pointLabel
      };
      setDraftPrompts((current) => [...current, prompt]);
      return;
    }
    setPreviewObjectId(null);
    setDraftBox({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  }

  function handleStagePointerMove(event: Konva.KonvaEventObject<PointerEvent>) {
    if (!draftBox) {
      return;
    }
    const stage = event.target.getStage();
    if (!stage) {
      return;
    }
    const point = stagePoint(stage);
    if (!point) {
      return;
    }
    setDraftBox((current) => (current ? { ...current, x2: point.x, y2: point.y } : null));
  }

  function handleStagePointerUp() {
    if (!draftBox) {
      return;
    }
    const box = normalizeBox(draftBox, frameSize.width, frameSize.height);
    if (box) {
      setDraftPrompts((current) => [...current, box]);
    }
    setDraftBox(null);
  }

  function removePrompt(index: number) {
    setPreviewObjectId(null);
    setDraftPrompts((current) => current.filter((_, promptIndex) => promptIndex !== index));
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Veraser</h1>
          <span>Video object removal demo</span>
        </div>
        <button className="primary-button" disabled={busy} onClick={() => fileInputRef.current?.click()}>
          {busy ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
          Upload video
        </button>
        <input
          ref={fileInputRef}
          className="hidden-input"
          type="file"
          accept="video/mp4,video/quicktime,video/x-matroska,video/x-msvideo,video/webm"
          onChange={(event) => void handleUpload(event.target.files?.[0])}
        />
      </header>

      {error ? (
        <div className="error-banner">
          <X size={18} />
          {error}
        </div>
      ) : null}

      <section className="workspace">
        <aside className="side-panel">
          <div className="panel-section">
            <h2>Session</h2>
            <dl className="meta-list">
              <div>
                <dt>Video</dt>
                <dd>{session?.video.filename ?? "No upload"}</dd>
              </div>
              <div>
                <dt>Frame</dt>
                <dd>{session ? `${frameSize.width} x ${frameSize.height}` : "-"}</dd>
              </div>
              <div>
                <dt>Objects</dt>
                <dd>{objects.length}</dd>
              </div>
            </dl>
            <SpeedMetrics title="Upload Speed" metrics={session?.performance ?? []} />
          </div>

          <div className="panel-section">
            <div className="section-heading">
              <h2>Tools</h2>
              <button className="icon-button" title="New object" onClick={startNewObject}>
                <Eraser size={17} />
              </button>
            </div>
            <div className="segmented">
              <button className={tool === "box" ? "active" : ""} onClick={() => setTool("box")} title="Bounding box">
                <BoxSelect size={17} />
                Box
              </button>
              <button className={tool === "point" ? "active" : ""} onClick={() => setTool("point")} title="Point prompt">
                <CircleDot size={17} />
                Point
              </button>
              <button className={tool === "select" ? "active" : ""} onClick={() => setTool("select")} title="Select">
                <MousePointer2 size={17} />
                Select
              </button>
            </div>
            <div className="segmented compact">
              <button
                className={pointLabel === "foreground" ? "active positive" : ""}
                onClick={() => setPointLabel("foreground")}
              >
                Foreground
              </button>
              <button
                className={pointLabel === "background" ? "active negative" : ""}
                onClick={() => setPointLabel("background")}
              >
                Background
              </button>
            </div>
          </div>

          <div className="panel-section">
            <div className="section-heading">
              <h2>{selectedObject ? "Edit Object" : "New Object"}</h2>
              <button className="primary-button small" disabled={!session || draftPrompts.length === 0 || busy} onClick={() => void saveDraft()}>
                <Check size={16} />
                Save
              </button>
            </div>
            <div className="prompt-list">
              {draftPrompts.length === 0 ? <span className="empty-state">No prompts</span> : null}
              {draftPrompts.map((prompt, index) => (
                <div className="prompt-row" key={promptKey(prompt, index)}>
                  <span>{promptSummary(prompt)}</span>
                  <button className="icon-button" title="Remove prompt" onClick={() => removePrompt(index)}>
                    <X size={15} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="panel-section">
            <div className="section-heading">
              <h2>Objects</h2>
              <button className="primary-button small" disabled={!session || objects.length === 0 || busy} onClick={() => void runMasking()}>
                <Play size={16} />
                Trace
              </button>
            </div>
            <div className="object-list">
              {objects.length === 0 ? <span className="empty-state">No saved objects</span> : null}
              {objects.map((object, index) => (
                <div className={object.object_id === selectedObjectId ? "object-row selected" : "object-row"} key={object.object_id}>
                  <button onClick={() => selectObject(object)}>
                    <span>Object {index + 1}</span>
                    <small>{object.prompts.length} prompts</small>
                  </button>
                  <button
                    className="secondary-button small"
                    title="Preview first-frame mask"
                    onClick={() => previewObject(object)}
                  >
                    <Eye size={16} />
                    Preview
                  </button>
                  <button className="icon-button danger" title="Delete object" onClick={() => void removeObject(object.object_id)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <section className="viewer-panel">
          <div className="viewer-toolbar">
            <div>
              <strong>{session ? "First frame" : "Upload a video"}</strong>
              <span>{selectedObject ? selectedObject.object_id : "new object draft"}</span>
            </div>
            {job ? (
              <div className={`job-pill ${job.status}`}>
                {job.status === "running" || job.status === "pending" ? <Loader2 className="spin" size={15} /> : null}
                {job.current_stage} · {job.frames_done}/{job.frames_total ?? "-"}
              </div>
            ) : null}
            {inpaintingJob ? (
              <div className={`job-pill ${inpaintingJob.status}`}>
                {inpaintingJob.status === "running" || inpaintingJob.status === "pending" ? <Loader2 className="spin" size={15} /> : null}
                inpaint: {inpaintingJob.current_stage} · {inpaintingJob.frames_done}/{inpaintingJob.frames_total ?? "-"}
              </div>
            ) : null}
          </div>

          <div className="canvas-host" ref={canvasHostRef}>
            <Stage
              width={stageWidth}
              height={stageHeight}
              className="annotation-stage"
              onPointerDown={handleStagePointerDown}
              onPointerMove={handleStagePointerMove}
              onPointerUp={handleStagePointerUp}
            >
              <Layer scaleX={scale} scaleY={scale}>
                {firstFrameImage ? (
                  <KonvaImage image={firstFrameImage} width={frameSize.width} height={frameSize.height} listening={false} />
                ) : (
                  <Rect width={frameSize.width} height={frameSize.height} fill="#20252b" listening={false} />
                )}
                {trackedMaskImage ? (
                  <KonvaImage image={trackedMaskImage} width={frameSize.width} height={frameSize.height} opacity={0.45} listening={false} />
                ) : null}
                {previewImage ? (
                  <KonvaImage image={previewImage} width={frameSize.width} height={frameSize.height} opacity={0.85} listening={false} />
                ) : null}
                <Group>
                  {visiblePrompts.map((prompt, index) =>
                    prompt.type === "point" ? (
                      <Circle
                        key={promptKey(prompt, index)}
                        x={prompt.x}
                        y={prompt.y}
                        radius={7}
                        fill={prompt.label === "foreground" ? "#31c48d" : "#f05252"}
                        stroke="#ffffff"
                        strokeWidth={2}
                      />
                    ) : (
                      <Rect
                        key={promptKey(prompt, index)}
                        x={prompt.x1}
                        y={prompt.y1}
                        width={prompt.x2 - prompt.x1}
                        height={prompt.y2 - prompt.y1}
                        stroke="#f9c74f"
                        strokeWidth={3}
                        dash={[8, 5]}
                      />
                    )
                  )}
                  {draftBox ? (
                    <Rect
                      x={Math.min(draftBox.x1, draftBox.x2)}
                      y={Math.min(draftBox.y1, draftBox.y2)}
                      width={Math.abs(draftBox.x2 - draftBox.x1)}
                      height={Math.abs(draftBox.y2 - draftBox.y1)}
                      stroke="#f9c74f"
                      strokeWidth={2}
                    />
                  ) : null}
                </Group>
              </Layer>
            </Stage>
          </div>

          <div className="video-preview-strip">
            <section className="video-preview">
              <div>
                <strong>Source video</strong>
                <span>{session?.video.filename ?? "No upload"}</span>
              </div>
              {sourceVideoUrl ? (
                <video key={sourceVideoUrl} controls preload="metadata" src={sourceVideoUrl} />
              ) : (
                <div className="video-placeholder">No video</div>
              )}
            </section>
            <section className="video-preview">
              <div>
                <strong>Processed video</strong>
                <span>{inpaintingJob?.status === "succeeded" ? "Ready" : "Not ready"}</span>
              </div>
              {processedVideoUrl ? (
                <video key={processedVideoUrl} controls preload="metadata" src={processedVideoUrl} />
              ) : (
                <div className="video-placeholder">No output</div>
              )}
            </section>
          </div>

          <footer className="result-strip">
            <div>
              <strong>Tracked mask</strong>
              <span>{manifest ? `${manifest.frames_total} frames` : "No manifest"}</span>
            </div>
            <div className="speed-summary">
              <SpeedMetrics title="Trace Speed" metrics={job?.performance ?? []} />
              <SpeedMetrics title="Inpaint Speed" metrics={inpaintingJob?.performance ?? []} />
            </div>
            <label>
              Frame
              <input
                type="number"
                min={0}
                max={Math.max((manifest?.frames_total ?? 1) - 1, 0)}
                value={maskFrameIndex}
                onChange={(event) => setMaskFrameIndex(Number(event.target.value))}
              />
            </label>
            {session && job?.status === "succeeded" && inpaintingJob?.status !== "succeeded" ? (
              <button className="primary-button small" disabled={busy || inpaintingJob?.status === "running" || inpaintingJob?.status === "pending"} onClick={() => void runInpainting()}>
                {inpaintingJob?.status === "running" || inpaintingJob?.status === "pending" ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
                Inpaint
              </button>
            ) : null}
            {inpaintingJob?.status === "succeeded" ? (
              <a className="secondary-button" href={processedVideoUrl ?? "#"} target="_blank" rel="noreferrer">
                Processed video
              </a>
            ) : null}
          </footer>
        </section>
      </section>
    </main>
  );
}
