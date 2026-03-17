"""Predefined team member knowledge bases."""

TEAM = {
    "alice": {
        "role": "Frontend Engineer",
        "snippets": [
            "We use React 18 with TypeScript for all frontend components",
            "The design system is built on Tailwind CSS with custom tokens",
            "State management uses Zustand for local state and React Query for server state",
            "Our component library lives in packages/ui and uses Storybook for documentation",
            "We run Playwright end-to-end tests against staging before each release",
            "The frontend build pipeline uses Vite with code splitting per route",
            "Accessibility is enforced with eslint-plugin-jsx-a11y and axe-core in CI",
            "We use React Router v6 with lazy loading for all page-level routes",
            "Performance budgets are set at 200KB initial JS and 3s LCP on 3G",
            "Our error boundary component reports to Sentry with user context",
        ],
    },
    "bob": {
        "role": "Backend Engineer",
        "snippets": [
            "The API is built with FastAPI and uses Pydantic v2 for validation",
            "We use PostgreSQL 16 with SQLAlchemy 2.0 async ORM",
            "Authentication uses JWT tokens with 15-minute access and 7-day refresh",
            "Background jobs run on Celery with Redis as the broker",
            "API rate limiting is handled by a Redis sliding window at 100 req/min",
            "Database migrations use Alembic with auto-generated revision scripts",
            "We implement the repository pattern to keep business logic separate from data access",
            "The API follows REST conventions with versioning via URL prefix /api/v1",
            "Structured logging goes to stdout in JSON format, collected by Fluentd",
            "Integration tests use testcontainers to spin up real Postgres instances",
        ],
    },
    "carol": {
        "role": "DevOps Engineer",
        "snippets": [
            "Infrastructure is defined in Terraform with remote state in S3",
            "Kubernetes runs on EKS with node groups sized for memory-intensive workloads",
            "CI/CD uses GitHub Actions with separate workflows for test, build, and deploy",
            "Docker images are built with multi-stage builds and pushed to ECR",
            "Monitoring stack is Prometheus, Grafana, and Alertmanager with PagerDuty",
            "Secrets management uses AWS Secrets Manager with automatic rotation",
            "We deploy to staging automatically on merge to main, production requires approval",
            "Database backups run hourly via pg_dump to S3 with 30-day retention",
            "Horizontal pod autoscaling is configured for CPU and memory targets",
            "SSL certificates are managed by cert-manager with Let's Encrypt",
        ],
    },
    "dave": {
        "role": "Security Engineer",
        "snippets": [
            "We run Snyk scans on every PR to catch vulnerable dependencies",
            "OWASP ZAP runs nightly against staging for dynamic security testing",
            "All secrets are rotated every 90 days using automated rotation policies",
            "We enforce MFA for all production access including CI service accounts",
            "Container images are scanned with Trivy before deployment",
            "Network policies restrict pod-to-pod communication to explicit allow rules",
            "Audit logs are immutable and shipped to a separate security account",
            "We conduct quarterly penetration tests with external security firms",
        ],
    },
}

QUERIES = [
    "How do we build and bundle the frontend application?",
    "What database do we use and how are migrations handled?",
    "How is the application deployed to production?",
    "How do we handle user authentication and tokens?",
    "What monitoring and alerting tools do we use?",
    "How do we run end-to-end tests?",
    "How do we manage secrets and credentials?",
    "How do we handle API rate limiting?",
]

QUERIES_WITH_DAVE = [
    "How do we scan for security vulnerabilities?",
    "How is the application deployed to production?",
    "How do we manage secrets and credentials?",
    "What security testing do we perform?",
]
