import { Container, SimpleGrid, Skeleton, Stack } from "@mantine/core";

export default function Loading() {
  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Skeleton height={340} radius="xl" />
        <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
          <Skeleton height={360} radius="xl" />
          <Skeleton height={360} radius="xl" />
        </SimpleGrid>
        <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="lg">
          <Skeleton height={460} radius="xl" />
          <Skeleton height={460} radius="xl" />
        </SimpleGrid>
      </Stack>
    </Container>
  );
}
