"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import {
  Button,
  Card,
  FileInput,
  Group,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { Controller, useForm } from "react-hook-form";

import {
  pipelineFormSchema,
  type PipelineFormValues,
} from "@/lib/forms/schemas";

type PipelineFormProps = {
  isSubmitting: boolean;
  onSubmit: (values: PipelineFormValues) => void;
};

export function PipelineForm({
  isSubmitting,
  onSubmit,
}: PipelineFormProps) {
  const {
    handleSubmit,
    control,
    formState: { errors },
  } = useForm<PipelineFormValues>({
    resolver: zodResolver(pipelineFormSchema),
    defaultValues: {
      reportFile: undefined,
    },
    mode: "onSubmit",
  });

  return (
    <Card withBorder radius="xl" shadow="sm" padding="lg">
      <Stack>
        <Title order={3}>Pipeline Analyzer</Title>
        <Text size="sm" c="dimmed">
          Upload your blood report and get summary with recommendations in one click.
        </Text>

        <Controller
          control={control}
          name="reportFile"
          render={({ field }) => {
            const inputValue = field.value instanceof File ? field.value : null;
            return (
              <FileInput
                label="Blood Report File"
                description="Upload PDF or image report for end-to-end analysis"
                placeholder="Select report file"
                accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp"
                value={inputValue}
                onChange={(value) => field.onChange(value ?? undefined)}
                error={errors.reportFile?.message}
              />
            );
          }}
        />

        <Group justify="flex-end">
          <Button loading={isSubmitting} onClick={handleSubmit(onSubmit)}>
            Generate Summary and Recommendations
          </Button>
        </Group>
      </Stack>
    </Card>
  );
}
