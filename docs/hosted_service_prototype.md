# Hosted Service Prototype Architecture

This document outlines a pragmatic in-house hosting approach for the Gitview analysis pipeline using a containerized service catalog, per-analysis isolation, and a JSON-capable database backend.

## Goals
- Host the analysis platform internally while preserving a path to production-hardening.
- Isolate each analysis in its own container with strict tenant boundaries.
- Persist analysis state in a document/database backend that supports JSON and per-tenant access control.
- Provide operational hooks for logging, scrubbing, and lifecycle automation.

## Core services
- **Service catalog**: Registry describing available analysis profiles (e.g., full history, incremental) and model/runtime versions. Enables controlled rollout and A/B testing of hosted model upgrades.
- **Instance tracker**: Tracks container lifecycle per analysis run with audit metadata (account, repo fingerprint, branch, commits analyzed, timestamps). Supports restart policies and TTL-based cleanup.
- **Analysis runner**: Spins up short-lived containers to execute the extract→enrich→chunk→summarize→story→write pipeline. Receives signed credentials (LLM, GitHub, DB) at start, mounted via ephemeral secrets.
- **Database gateway**: Thin service that mediates access to the JSON-capable backend (e.g., Postgres JSONB, MongoDB). Handles multi-tenant scoping, optimistic locking for incremental runs, and points large artifacts to object storage.

## Data model outline
- **Accounts**: Tenant metadata, credential references, retention settings, and policy flags (PII scrubbing required, review data allowed, etc.).
- **Repositories**: Fingerprint of the git remote/branch plus last analyzed commit/date. Used to prevent duplicate runs and to gate incremental updates.
- **Analyses**: Stored as JSON documents containing `AnalysisConfig`, `AnalysisContext` summary, handler metrics, model versions, and status.
- **Phases and commits**: Derived artifacts (summaries, timelines, recommendations) stored as nested JSON; long markdown/timeline outputs streamed to object storage with object keys saved in the document.

## Security and isolation
- Per-analysis containers run with minimal permissions and no shared filesystem; all state is persisted through the DB gateway.
- Secrets are injected at startup (short TTL) and never written to disk. Audit logs capture access and mutation events per tenant.
- PII/secret scrubbing runs as a post-processing step before persistence and again on retrieval, based on account policies.

## Operational flow
1. Client requests an analysis via the service catalog with account context and repo parameters.
2. Instance tracker records the request and launches an analysis runner container.
3. Runner executes the pipeline and writes outputs through the database gateway.
4. Tracker marks completion, emits metrics, and schedules cleanup of the container and any temporary volumes.
5. Downstream hosted-model services can fetch persisted JSON slices to generate strategy or code-quality recommendations without reprocessing git history.

## Observability and resilience
- Centralized logging per analysis ID, with structured fields for account, repo fingerprint, and handler stage.
- Metrics on container start latency, handler duration, DB write/read throughput, and model token usage.
- Retry policies at the gateway for transient DB errors; idempotent upserts keyed by (account, repo, branch, commit range).
- Background janitors enforce retention, purge expired secrets, and reclaim orphaned containers.

## Startup scripts (high level)
- Bootstrap script to register service catalog entries and seed model/runtime versions.
- Launch script for the database gateway (configuring allowed tenants, indexes, and object storage bucket bindings).
- Runner entrypoint script that accepts signed config, hydrates the analysis pipeline, and streams outputs to the gateway before teardown.

This prototype keeps the codebase container-friendly while introducing the database backend and control-plane pieces needed for an internal hosted service.
