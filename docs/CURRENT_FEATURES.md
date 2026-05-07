# Current Feature Inventory

## Repository

- Name: `AI-SDK-CREWAI`
- SDK: CrewAI
- Positioning: Role-based autonomous crews for delegated mission execution.

## Implemented Today

- Agents Army routing with skill-aware primary agent selection.
- FastAPI service and local CLI entrypoint.
- CrewAI crew scaffold creation with graceful runtime-config messaging.
- Docker packaging, CI workflow, and pytest contract tests.
- Portfolio metadata, strategy notes, and skill matrix.

## Not Yet Implemented

- Add specialized crew templates for engineering, security, and deployment missions.
- Connect tool permissions and execution guardrails.
- Persist task outputs and verification evidence.

## Verification Contract

- The local runner must complete without crashing when optional SDK credentials are missing.
- The API contract must return routing and verification fields.
- Tests must prove mission routing and a security-focused SENTINEL route.
