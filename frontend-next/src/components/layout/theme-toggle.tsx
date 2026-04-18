"use client";

import {
  ActionIcon,
  Tooltip,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";
import { useMounted } from "@mantine/hooks";
import { motion } from "framer-motion";
import { Moon, SunMedium } from "lucide-react";

export function ThemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  const computedScheme = useComputedColorScheme("light", {
    getInitialValueInEffect: true,
  });
  const mounted = useMounted();

  const isDark = mounted && computedScheme === "dark";

  return (
    <Tooltip label={isDark ? "Switch to light" : "Switch to dark"}>
      <ActionIcon
        variant="light"
        size="lg"
        radius="xl"
        aria-label="Toggle color scheme"
        onClick={() => setColorScheme(isDark ? "light" : "dark")}
      >
        <motion.span
          key={isDark ? "dark" : "light"}
          initial={{ rotate: -120, opacity: 0, scale: 0.8 }}
          animate={{ rotate: 0, opacity: 1, scale: 1 }}
          transition={{ type: "spring", stiffness: 170, damping: 16 }}
        >
          {isDark ? <SunMedium size={17} /> : <Moon size={17} />}
        </motion.span>
      </ActionIcon>
    </Tooltip>
  );
}
