import { z } from "zod";

export const statusSummarySchema = z.object({
  low: z.number().int().nonnegative(),
  normal: z.number().int().nonnegative(),
  high: z.number().int().nonnegative(),
  unknown: z.number().int().nonnegative(),
});

export const nerEntitySchema = z.object({
  label: z.string(),
  text: z.string(),
  start: z.number().int().nonnegative(),
  end: z.number().int().nonnegative(),
  score: z.number().min(0).max(1),
});

export const interpretedRowSchema = z
  .object({
    biomarker: z.string().optional(),
    biomarker_normalized: z.string().optional(),
    status: z.string().optional(),
    value: z.union([z.number(), z.string()]).optional(),
    unit: z.string().optional(),
    reference_range: z.string().optional(),
  })
  .passthrough();

export const recommendationItemSchema = z
  .object({
    id: z.string(),
    text: z.string(),
    source: z.string().optional(),
    score: z.number().optional(),
    combined_score: z.number().optional(),
    method: z.string().optional(),
    methods: z.array(z.string()).optional(),
  })
  .passthrough();

export const pipelineResponseSchema = z
  .object({
    document_id: z.string(),
    source_path: z.string(),
    extraction: z.record(z.string(), z.unknown()),
    ner: z.object({
      entity_count: z.number().int().nonnegative(),
      entities: z.array(nerEntitySchema),
    }),
    interpretation: z.object({
      row_count: z.number().int().nonnegative(),
      status_summary: statusSummarySchema,
      rows: z.array(interpretedRowSchema),
    }),
    recommendation: z.object({
      query: z.string(),
      results: z.array(recommendationItemSchema),
      summary: z.string(),
    }),
  })
  .passthrough();

export const recommendationResponseSchema = z
  .object({
    query: z.string(),
    results: z.array(recommendationItemSchema),
    summary: z.string(),
  })
  .passthrough();

export const healthResponseSchema = z.object({
  status: z.string(),
});

export const apiErrorSchema = z
  .object({
    detail: z.union([z.string(), z.array(z.unknown())]).optional(),
  })
  .passthrough();

export type StatusSummary = z.infer<typeof statusSummarySchema>;
export type NerEntity = z.infer<typeof nerEntitySchema>;
export type InterpretedRow = z.infer<typeof interpretedRowSchema>;
export type RecommendationItem = z.infer<typeof recommendationItemSchema>;
export type PipelineResponse = z.infer<typeof pipelineResponseSchema>;
export type RecommendationResponse = z.infer<
  typeof recommendationResponseSchema
>;
export type HealthResponse = z.infer<typeof healthResponseSchema>;
