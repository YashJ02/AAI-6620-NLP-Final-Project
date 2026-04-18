# Blood Report Analyzer Frontend

Production-oriented Next.js App Router frontend for the Health Blood Report Analyzer stack.

## Stack

- Runtime/package manager: Bun
- Framework: Next.js 16 + React 19 + TypeScript
- UI library: Mantine
- Data fetching/cache: React Query
- State management: Zustand
- Validation/forms: Zod + React Hook Form
- Animation: Framer Motion
- Image optimization: Next Image + sharp blur placeholder pipeline

## Start

```bash
bun install
bun run dev
```

Open http://localhost:3000.

## Quality Gates

```bash
bun run check
bun run build
```

## Backend Dependency

The app expects FastAPI backend endpoints from this repository.

- Default base URL: `http://localhost:8000`
- Health endpoint: `GET /health`
- Main pipeline endpoint: `POST /v1/pipeline`
- Recommendation endpoint: `POST /v1/recommend`

## UX Notes

- Smooth light/dark theme transition with persistent preference
- Skeletons for loading states
- Route-level error and loading boundaries
- Typed API contracts and resilient runtime error messages
