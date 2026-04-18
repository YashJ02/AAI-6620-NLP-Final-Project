import { z } from "zod";

const backendUrlSchema = z
  .string()
  .trim()
  .refine((value) => {
    try {
      const parsed = new URL(value);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  }, "Enter a valid http/https URL.");

const reportFileSchema = z
  .custom<File>(
    (value) =>
      typeof File !== "undefined" && value instanceof File && value.size > 0,
    "Upload a report file."
  )
  .refine(
    (file) => /\.(pdf|png|jpg|jpeg|tif|tiff|bmp)$/i.test(file.name),
    "Only PDF or image files are supported."
  );

export const pipelineFormSchema = z.object({
  reportFile: reportFileSchema,
});

export const recommendationFormSchema = z.object({
  backendUrl: backendUrlSchema,
  query: z.string().trim().min(3, "Add at least 3 characters."),
  patientId: z.string().trim().min(2).max(60),
  topK: z.number().int().min(1).max(10),
});

export type PipelineFormValues = z.infer<typeof pipelineFormSchema>;
export type RecommendationFormValues = z.infer<typeof recommendationFormSchema>;
