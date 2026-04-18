import {
  apiErrorSchema,
  healthResponseSchema,
  pipelineResponseSchema,
  recommendationResponseSchema,
  type HealthResponse,
  type PipelineResponse,
  type RecommendationResponse,
} from "@/lib/api/schemas";
import { z } from "zod";

const recommendationRequestSchema = z.object({
  query: z.string().trim(),
  top_k: z.number().int().min(1).max(10),
  patient_id: z.string().trim().default("web-demo"),
});

const pipelineRequestSchema = z.object({
  pdf_path: z.string().trim().min(1),
  model_dir: z.string().trim().min(1),
});

const pipelineUploadRequestSchema = z.object({
  file: z.custom<File>(
    (value) =>
      typeof File !== "undefined" && value instanceof File && value.size > 0,
    "Report file is required."
  ),
  model_dir: z.string().trim().min(1),
});

export class ApiClientError extends Error {
  public readonly status: number;
  public readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
  }
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

async function parseJsonSafe(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

async function requestJson<T>(
  baseUrl: string,
  path: string,
  options: RequestInit,
  parser: (payload: unknown) => T
): Promise<T> {
  const headers = new Headers(options.headers ?? {});
  const isFormDataBody =
    typeof FormData !== "undefined" && options.body instanceof FormData;

  if (!isFormDataBody && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, {
      ...options,
      headers,
      cache: "no-store",
    });
  } catch (error) {
    throw new ApiClientError(
      "Could not connect to the analysis server. Please make sure backend is running.",
      0,
      error
    );
  }

  const payload = await parseJsonSafe(response);

  if (!response.ok) {
    const apiError = apiErrorSchema.safeParse(payload);
    const detail = apiError.success ? apiError.data.detail : payload;
    const message =
      typeof detail === "string"
        ? detail
        : `Request failed with status ${response.status}`;
    throw new ApiClientError(message, response.status, detail);
  }

  return parser(payload);
}

export async function fetchHealth(baseUrl: string): Promise<HealthResponse> {
  return requestJson(baseUrl, "/health", { method: "GET" }, (payload) =>
    healthResponseSchema.parse(payload)
  );
}

export async function runPipeline(
  baseUrl: string,
  input: z.input<typeof pipelineRequestSchema>
): Promise<PipelineResponse> {
  const body = pipelineRequestSchema.parse(input);
  return requestJson(
    baseUrl,
    "/v1/pipeline",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
    (payload) => pipelineResponseSchema.parse(payload)
  );
}

export async function runPipelineUpload(
  baseUrl: string,
  input: z.input<typeof pipelineUploadRequestSchema>
): Promise<PipelineResponse> {
  const body = pipelineUploadRequestSchema.parse(input);
  const formData = new FormData();
  formData.append("file", body.file);
  formData.append("model_dir", body.model_dir);

  return requestJson(
    baseUrl,
    "/v1/pipeline/upload",
    {
      method: "POST",
      body: formData,
    },
    (payload) => pipelineResponseSchema.parse(payload)
  );
}

export async function runRecommendation(
  baseUrl: string,
  input: z.input<typeof recommendationRequestSchema>
): Promise<RecommendationResponse> {
  const parsed = recommendationRequestSchema.parse(input);
  return requestJson(
    baseUrl,
    "/v1/recommend",
    {
      method: "POST",
      body: JSON.stringify({
        query: parsed.query,
        top_k: parsed.top_k,
        patient_id: parsed.patient_id,
        interpreted_rows: [],
        ner_entities: [],
        status_summary: {
          low: 0,
          normal: 0,
          high: 0,
          unknown: 0,
        },
      }),
    },
    (payload) => recommendationResponseSchema.parse(payload)
  );
}
