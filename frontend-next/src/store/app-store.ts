import {
  type PipelineResponse,
} from "@/lib/api/schemas";
import { create } from "zustand";

type AppStoreState = {
  pipelineResult: PipelineResponse | null;
  setPipelineResult: (value: PipelineResponse | null) => void;
  resetAll: () => void;
};

export const useAppStore = create<AppStoreState>((set) => ({
  pipelineResult: null,
  setPipelineResult: (value) => set({ pipelineResult: value }),
  resetAll: () =>
    set({
      pipelineResult: null,
    }),
}));
