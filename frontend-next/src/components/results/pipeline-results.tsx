import {
  Badge,
  Card,
  Divider,
  Group,
  List,
  ScrollArea,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
} from "@mantine/core";

import type { PipelineResponse } from "@/lib/api/schemas";

type PipelineResultsProps = {
  data: PipelineResponse;
};

function cleanBiomarkerText(text: string | undefined): string {
  if (!text) {
    return "";
  }
  return text
    .replace(/^[A-Za-z0-9]+\)\s*/u, "")
    .replace(/\s+/gu, " ")
    .replace(/[ :=<>\-.]+$/u, "")
    .trim();
}

function formatBiomarkerLabel(row: PipelineResponse["interpretation"]["rows"][number]): string {
  const normalized = cleanBiomarkerText(row.biomarker_normalized?.trim());
  if (normalized) {
    return normalized;
  }
  const primary = cleanBiomarkerText(row.biomarker?.trim());
  if (primary) {
    return primary;
  }
  return "Unknown marker";
}

function formatRangeText(row: PipelineResponse["interpretation"]["rows"][number]): string {
  const range = row.reference_range?.trim();
  if (range) {
    return range;
  }
  if ((row as Record<string, unknown>).range_source === "inferred") {
    return "Estimated from standard reference";
  }
  return "-";
}

function formatValueText(row: PipelineResponse["interpretation"]["rows"][number]): string {
  const value = row.value;
  const unit = row.unit?.trim();
  if (value === undefined || value === null || value === "") {
    return "Value unavailable";
  }
  return `${String(value)}${unit ? ` ${unit}` : ""}`;
}

function statusSentence(summary: PipelineResponse["interpretation"]["status_summary"]): string {
  if (summary.high + summary.low === 0) {
    return "No clearly abnormal markers were detected in the parsed table rows.";
  }
  if (summary.high > 0 && summary.low > 0) {
    return "Both high and low markers were detected. Focus on balancing the flagged values with your clinician.";
  }
  if (summary.high > 0) {
    return "High markers were detected. Prioritize diet and follow-up for these elevated values.";
  }
  return "Low markers were detected. Prioritize nutrition and recovery guidance for these reduced values.";
}

function statusTone(
  status: string | undefined
): "teal" | "red" | "blue" | "gray" {
  if (status === "high") {
    return "red";
  }
  if (status === "low") {
    return "blue";
  }
  if (status === "normal") {
    return "teal";
  }
  return "gray";
}

export function PipelineResults({ data }: PipelineResultsProps) {
  const meaningfulRows = data.interpretation.rows.filter(
    (row) =>
      row.status !== "unknown" ||
      Boolean(row.reference_range) ||
      (row as Record<string, unknown>).range_source === "inferred"
  );
  const rows = (meaningfulRows.length > 0 ? meaningfulRows : data.interpretation.rows).slice(0, 12);
  const summary = data.interpretation.status_summary;
  const flaggedRows = data.interpretation.rows
    .filter((row) => row.status === "high" || row.status === "low")
    .slice(0, 6);

  const recommendationCards = data.recommendation.results.slice(0, 5);
  const fallbackRecommendations = [
    "Hydrate well and repeat blood work under consistent fasting conditions.",
    "Track the same biomarkers over time rather than one isolated result.",
    "Discuss elevated or borderline values with your clinician before major diet or supplement changes.",
  ];
  const renderedRecommendations =
    recommendationCards.length > 0
      ? recommendationCards
      : fallbackRecommendations.map((text, idx) => ({
          id: `fallback-${idx}`,
          text,
          combined_score: undefined,
        }));

  const nextSteps: string[] = [];
  if (summary.high + summary.low > 0) {
    nextSteps.push("Review flagged markers with your doctor and compare with your symptoms.");
    nextSteps.push("Repeat testing if values were unexpected or from a non-fasting sample.");
  } else {
    nextSteps.push("Keep current habits and continue periodic monitoring.");
  }
  nextSteps.push("Follow the suggestions below as supportive lifestyle guidance.");
  nextSteps.push("If symptoms persist, seek medical review even when values are near normal.");

  return (
    <Stack>
      <Card withBorder radius="xl" shadow="sm" padding="lg">
        <Stack gap="sm">
          <Title order={3}>Analysis Snapshot</Title>
          <Text size="sm" c="dimmed">
            Source: {data.source_path}
          </Text>
          <SimpleGrid cols={{ base: 2, md: 4 }}>
            <Card withBorder radius="md" padding="sm">
              <Text size="xs" c="dimmed">
                NER Entities
              </Text>
              <Text fw={700} size="xl">
                {data.ner.entity_count}
              </Text>
            </Card>
            <Card withBorder radius="md" padding="sm">
              <Text size="xs" c="dimmed">
                High
              </Text>
              <Text fw={700} size="xl" c="red.6">
                {summary.high}
              </Text>
            </Card>
            <Card withBorder radius="md" padding="sm">
              <Text size="xs" c="dimmed">
                Low
              </Text>
              <Text fw={700} size="xl" c="blue.6">
                {summary.low}
              </Text>
            </Card>
            <Card withBorder radius="md" padding="sm">
              <Text size="xs" c="dimmed">
                Normal
              </Text>
              <Text fw={700} size="xl" c="teal.6">
                {summary.normal}
              </Text>
            </Card>
          </SimpleGrid>
        </Stack>
      </Card>

      <Card withBorder radius="xl" shadow="sm" padding="lg">
        <Stack>
          <Title order={3}>Interpreted Biomarkers</Title>
          <Text size="sm" c="dimmed">
            Showing the most meaningful parsed rows from your report.
          </Text>
          <ScrollArea>
            <Table striped withTableBorder highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Biomarker</Table.Th>
                  <Table.Th>Value</Table.Th>
                  <Table.Th>Unit</Table.Th>
                  <Table.Th>Range</Table.Th>
                  <Table.Th>Status</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {rows.map((row, index) => (
                  <Table.Tr
                    key={`${formatBiomarkerLabel(row)}-${index}`}
                  >
                    <Table.Td>{formatBiomarkerLabel(row)}</Table.Td>
                    <Table.Td>{formatValueText(row)}</Table.Td>
                    <Table.Td>{row.unit ?? "-"}</Table.Td>
                    <Table.Td>{formatRangeText(row)}</Table.Td>
                    <Table.Td>
                      <Badge variant="light" color={statusTone(row.status)}>
                        {row.status ?? "unknown"}
                      </Badge>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        </Stack>
      </Card>

      <Card withBorder radius="xl" shadow="sm" padding="lg">
        <Stack>
          <Group justify="space-between">
            <Title order={3}>What This Means For You</Title>
            <Badge variant="dot" color="amber">
              Personalized guidance
            </Badge>
          </Group>
          <Text>{statusSentence(summary)}</Text>

          {flaggedRows.length > 0 ? (
            <Stack gap="xs">
              <Title order={5}>Most important flagged markers</Title>
              <Group>
                {flaggedRows.map((row, idx) => (
                  <Badge key={`${formatBiomarkerLabel(row)}-${idx}`} color={statusTone(row.status)} variant="light">
                    {formatBiomarkerLabel(row)}: {formatValueText(row)} ({row.status})
                  </Badge>
                ))}
              </Group>
            </Stack>
          ) : null}

          <Divider />

          <Title order={5}>Summary</Title>
          <Text>{data.recommendation.summary}</Text>

          <Text size="sm" c="dimmed">
            Knowledge query used: {data.recommendation.query || "No query generated."}
          </Text>

          <Title order={4}>Top Actionable Recommendations</Title>
          <SimpleGrid cols={{ base: 1, md: 2 }}>
            {renderedRecommendations.map((item) => (
              <Card key={item.id} withBorder radius="md" padding="md">
                <Stack gap="xs">
                  <Group justify="space-between">
                    <Badge color="teal" variant="light">
                      Suggestion
                    </Badge>
                    {typeof item.combined_score === "number" ? (
                      <Badge color="gray" variant="outline">
                        score {item.combined_score.toFixed(2)}
                      </Badge>
                    ) : null}
                  </Group>
                  <Text size="sm">{item.text}</Text>
                </Stack>
              </Card>
            ))}
          </SimpleGrid>

          <Title order={5}>Suggested next steps</Title>
          <List spacing="xs">
            {nextSteps.map((step, idx) => (
              <List.Item key={`step-${idx}`}>
                <Text size="sm">{step}</Text>
              </List.Item>
            ))}
          </List>
        </Stack>
      </Card>
    </Stack>
  );
}
