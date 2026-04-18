"use client";

import { Button, Card, Container, Stack, Text, Title } from "@mantine/core";

type ErrorProps = {
  error: Error & { digest?: string };
  reset: () => void;
};

export default function GlobalError({ error, reset }: ErrorProps) {
  return (
    <Container size="sm" py="xl">
      <Card withBorder radius="xl" padding="xl">
        <Stack>
          <Title order={2}>Something went wrong</Title>
          <Text c="dimmed">
            {error.message || "Unexpected rendering failure."}
          </Text>
          <Button onClick={reset}>Retry</Button>
        </Stack>
      </Card>
    </Container>
  );
}
