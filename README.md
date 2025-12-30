# FashionFlow Commerce Platform

Production-grade reference implementation for a fashion e-commerce brand that blends a LangGraph-powered recommendation engine, modern FastAPI backend, and cloud-native deployment tooling (Docker, Kubernetes, Nginx).

## Highlights
- **Multi-agent recommendation graph** built with LangGraph orchestrating trend research, stylist reasoning, and merchandising validation agents on top of a vector store of catalog assets.
- **FastAPI service layer** exposing catalog, cart, and recommendation endpoints with typed schemas, validation, and observability hooks.
- **Deployment-ready assets** including Dockerfile, docker-compose stack, Kubernetes manifests (deployment, service, ingress), and an edge Nginx reverse proxy tuned for blue/green rollouts.
- **CI/CD blueprints** for both GitHub Actions and GitLab CI with dev/test/prod branch protections, automated tests, security scanning, and canary deploy gates.
- **Environment-aware configuration** (12-factor) using Pydantic Settings to keep secrets in env vars and support Azure OpenAI or vanilla OpenAI backends.

## Repository Layout
```
.
├── README.md
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── catalog.py
│   │   │       ├── health.py
│   │   │       └── recommendations.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── data/
│   │   │   └── products.json
│   │   ├── services/
│   │   │   ├── agents.py
│   │   │   ├── catalog.py
│   │   │   ├── graph.py
│   │   │   └── vectorstore.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── pyproject.toml
│   ├── uv.lock (generated)
│   └── tests/
│       └── test_recommendations.py
├── docker/
│   └── Dockerfile
├── docker-compose.yaml
├── k8s/
│   ├── configmap.yaml
│   ├── deployment.yaml
│   ├── ingress.yaml
│   └── service.yaml
├── nginx/
│   └── default.conf
├── .github/workflows/ci.yml
├── .gitlab-ci.yml
└── rag_script.py (standalone ingestion helper)
```

## Multi-Agent Recommendation Flow
The LangGraph pipeline operates on a shared `RecommendationState` TypedDict:
1. **TrendScout agent** queries the vector store (Chroma or pgvector) for current season trends plus metadata from analytics feeds.
2. **Stylist agent** blends user profile (fit, palette, context) with trend signals using GPT-4o or Azure GPT-4 Turbo to reason about outfits and style notes.
3. **Merchandiser agent** validates stock, margin, and regional restrictions before finalizing the shortlist, ensuring every suggestion is sellable.
4. **Edge summarizer** generates human-friendly copy, CTA messaging, and recommended bundle pricing.

LangGraph composes these agents with guardrails (input validation, telemetry) and supports streaming outputs for conversational UX.

## Local Development
1. **Install dependencies** (recommended via uv or pip):
   ```bash
   cd backend
   uv sync  # or: pip install -r requirements.txt generated from pyproject
   ```
2. **Environment variables** (place in `.env` or export):
   ```bash
   export OPENAI_API_KEY=sk-...
   export LLM_PROVIDER=openai  # options: openai, azure, mock
   export VECTORSTORE_PATH=.data/chroma
   ```
3. **Run FastAPI**:
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Request recommendations**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/recommendations \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": "u123",
           "style_goals": ["evening", "minimal"],
           "preferred_colors": ["black", "gold"],
           "budget": 350
         }'
   ```

## Containerization & Reverse Proxy
- **Docker**: `docker build -t fashionflow-api -f docker/Dockerfile .`
- **docker-compose**: `docker compose up api nginx vectorstore`
- **Nginx** terminates TLS, adds caching headers, and forwards `/api` to FastAPI while serving static marketing assets from `/usr/share/nginx/html`.

## Kubernetes (prod/test)
Apply manifests (adjust namespace, secrets, storage classes):
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```
Key features:
- Rolling strategy with maxSurge 1 / maxUnavailable 0.
- ConfigMap-driven feature flags.
- Liveness/readiness probes wired to `/health/live` and `/health/ready`.
- HPA-ready resource requests/limits.
- Ingress compatible with AWS ALB, GKE, or Nginx Controller.

## CI/CD & Branching Strategy
- **Branches**: `dev` (feature integration), `test` (release candidates), `prod` (immutable releases). Protect `test`/`prod` with required reviews and passing checks.
- **GitHub Actions** (`.github/workflows/ci.yml`): lint → unit tests → security scan (Bandit, Trivy) → build/push Docker → deploy to dev namespace (optional auto).
- **GitLab CI** (`.gitlab-ci.yml`): mirrors stages (build, test, security, review, canary, prod). Uses environments tab for deploy tracking.
- **Versioning**: Semantic tags `vMAJOR.MINOR.PATCH`, automatically generated release notes.

## Observability & Ops
- Structured JSON logging with correlation IDs for every request.
- OpenTelemetry hooks for traces/metrics (insert collector endpoint in config).
- Feature flags for A/B testing of recommendation strategies (`AGENT_STRATEGY=multi_agent|rule_based`).
- Disaster recovery: daily vector store snapshots via CronJob, warm standby cluster defined in IaC (not included).

## Next Steps
- Connect real inventory DB (Postgres + Prisma or SQLModel).
- Add user auth/SSO (Cognito/Auth0) and payment integrations.
- Extend LangGraph with feedback loop capturing post-purchase satisfaction and returns data.
- Wire full-stack storefront (Next.js / Remix) that consumes these APIs and streams recommendations in real time.
