"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import {
  createTheme,
  localStorageColorSchemeManager,
  MantineProvider,
} from "@mantine/core";
import { Notifications } from "@mantine/notifications";
import { useState } from "react";

const appTheme = createTheme({
  primaryColor: "teal",
  defaultRadius: "md",
  fontFamily: "var(--font-manrope)",
  fontFamilyMonospace: "var(--font-jetbrains-mono)",
  headings: {
    fontFamily: "var(--font-sora)",
    fontWeight: "700",
  },
  colors: {
    teal: [
      "#ecfffd",
      "#d1f9f5",
      "#a5f3ec",
      "#5ee7db",
      "#22d3c5",
      "#14b8a6",
      "#0f9a8d",
      "#117a71",
      "#145f59",
      "#154d49",
    ],
    amber: [
      "#fff8e8",
      "#ffefcc",
      "#ffdf9a",
      "#ffd06c",
      "#ffc245",
      "#ffb42e",
      "#f0a11f",
      "#d28714",
      "#a96b0f",
      "#83540f",
    ],
  },
});

const colorSchemeManager = localStorageColorSchemeManager({
  key: "blood-report-theme",
});

type AppProvidersProps = {
  children: React.ReactNode;
};

export function AppProviders({ children }: AppProvidersProps) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            gcTime: 10 * 60_000,
            refetchOnWindowFocus: false,
            retry: 2,
          },
          mutations: {
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <MantineProvider
        theme={appTheme}
        defaultColorScheme="auto"
        colorSchemeManager={colorSchemeManager}
      >
        <Notifications position="top-right" zIndex={1200} />
        {children}
      </MantineProvider>
      <ReactQueryDevtools initialIsOpen={false} buttonPosition="bottom-left" />
    </QueryClientProvider>
  );
}
