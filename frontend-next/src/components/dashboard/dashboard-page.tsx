"use client";

import {
  Alert,
  Badge,
  Card,
  Container,
  Group,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AlertTriangle, Microscope } from "lucide-react";
import Image from "next/image";

import { AnalysisSkeleton } from "@/components/feedback/loading-skeletons";
import { PipelineForm } from "@/components/forms/pipeline-form";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { PipelineResults } from "@/components/results/pipeline-results";
import { ApiClientError, runPipelineUpload } from "@/lib/api/client";
import { type PipelineFormValues } from "@/lib/forms/schemas";
import { useAppStore } from "@/store/app-store";

type DashboardPageProps = {
  heroBlurDataURL: string;
};

const sectionTransition = {
  duration: 0.6,
  ease: "easeOut" as const,
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

function resolveErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) {
    return `${error.message} (HTTP ${error.status})`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected request failure.";
}

export function DashboardPage({ heroBlurDataURL }: DashboardPageProps) {
  const { pipelineResult, setPipelineResult } = useAppStore();

  const pipelineMutation = useMutation({
    mutationFn: async (values: PipelineFormValues) => {
      return runPipelineUpload(API_BASE_URL, {
        file: values.reportFile,
        model_dir: "artifacts/models/pubmedbert_ner/model",
      });
    },
    onSuccess: (data) => {
      setPipelineResult(data);
      notifications.show({
        title: "Pipeline complete",
        message: `Processed ${data.document_id} with ${data.ner.entity_count} entities.`,
        color: "teal",
      });
    },
    onError: (error) => {
      notifications.show({
        title: "Pipeline failed",
        message: resolveErrorMessage(error),
        color: "red",
      });
    },
  });
  const hasError = pipelineMutation.isError;
  const currentError = pipelineMutation.error;

  return (
    <Container size="xl" py="xl" className="dashboard-root">
      <Stack gap="xl">
        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={sectionTransition}
        >
          <Card withBorder radius="xl" className="hero-card" padding="xl">
            <Group align="stretch" wrap="wrap" justify="space-between">
              <Stack justify="center">
                <Group justify="space-between" align="center">
                  <Badge variant="dot" color="amber" size="lg">
                    Simple Health Insights
                  </Badge>
                  <ThemeToggle />
                </Group>

                <Title order={1} className="hero-title">
                  Upload Your Blood Report, Get Clear Guidance
                </Title>

                <Text size="lg" c="dimmed">
                  This tool reads your report and gives you an easy-to-understand
                  summary plus practical recommendations.
                </Text>

                <Badge variant="light" color="teal" w="fit-content">
                  One upload, one complete result
                </Badge>
              </Stack>

              <Card
                withBorder
                radius="lg"
                className="hero-image-wrap"
                padding={0}
                miw={280}
                maw={460}
                style={{ flex: "1 1 320px" }}
              >
                <Image
                  src="/images/hero-lab.jpg"
                  alt="Modern hematology laboratory dashboard"
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  placeholder="blur"
                  blurDataURL={heroBlurDataURL}
                  priority
                  className="hero-image"
                />
              </Card>
            </Group>
          </Card>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...sectionTransition, delay: 0.08 }}
        >
          <PipelineForm
            isSubmitting={pipelineMutation.isPending}
            onSubmit={(values) => pipelineMutation.mutate(values)}
          />
        </motion.section>

        {pipelineMutation.isPending ? (
          <AnalysisSkeleton />
        ) : null}

        {hasError ? (
          <Alert
            color="red"
            radius="lg"
            icon={<AlertTriangle size={16} />}
            title="Request error"
          >
            {resolveErrorMessage(currentError)}
          </Alert>
        ) : null}

        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...sectionTransition, delay: 0.16 }}
        >
          {pipelineResult ? (
            <PipelineResults data={pipelineResult} />
          ) : (
            <Card withBorder radius="xl" padding="lg">
              <Stack justify="center">
                <Group>
                  <Microscope size={18} />
                  <Title order={4}>Your summary will appear here</Title>
                </Group>
                <Text size="sm" c="dimmed">
                  Upload a report to see your health summary and recommendations.
                </Text>
              </Stack>
            </Card>
          )}
        </motion.section>
      </Stack>
    </Container>
  );
}
