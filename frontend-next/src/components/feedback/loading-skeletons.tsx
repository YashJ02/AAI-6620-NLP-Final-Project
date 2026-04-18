import { Card, Skeleton, Stack } from "@mantine/core";

export function AnalysisSkeleton() {
  return (
    <Stack>
      <Card withBorder radius="lg" shadow="sm" padding="lg">
        <Skeleton height={24} width="35%" mb="md" />
        <Skeleton height={14} width="85%" mb={10} />
        <Skeleton height={14} width="70%" />
      </Card>

      <Card withBorder radius="lg" shadow="sm" padding="lg">
        <Skeleton height={18} width="25%" mb="md" />
        <Skeleton height={56} radius="md" mb="sm" />
        <Skeleton height={56} radius="md" mb="sm" />
        <Skeleton height={56} radius="md" />
      </Card>
    </Stack>
  );
}
