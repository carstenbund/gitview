# Example Project Goals File

This is an example goals/objectives file for use with GitView's critical examination mode.

## How to Use

```bash
gitview analyze --critical --todo examples/PROJECT_GOALS.md
```

The LLM will evaluate your commit history against these objectives to identify:
- What goals were accomplished
- What goals were missed or incomplete
- Where development effort diverged from stated priorities
- Technical debt and gaps in functionality

---

# Project Objectives - Q1 2025

## Core Features

- [ ] Implement user authentication system
  - OAuth2 integration with Google, GitHub
  - Session management
  - Password reset flow

- [ ] Add API rate limiting
  - 1000 requests/hour per user
  - Redis-based rate limiter
  - Proper HTTP 429 responses

- [ ] Build dashboard analytics
  - Real-time metrics display
  - Chart visualizations
  - Export to CSV functionality

## Technical Improvements

- [ ] Improve test coverage to 80%
  - Unit tests for all core modules
  - Integration tests for API endpoints
  - E2E tests for critical user flows

- [ ] Database migration
  - Migrate from SQLite to PostgreSQL
  - Add connection pooling
  - Implement database backups

- [ ] Performance optimization
  - API response time < 100ms (p95)
  - Implement caching layer
  - Optimize database queries

## Documentation

- [ ] Complete API documentation
  - OpenAPI/Swagger spec
  - Example requests/responses
  - Authentication guide

- [ ] Write deployment guide
  - Docker setup instructions
  - Environment configuration
  - Production checklist

## Security

- [ ] Security audit findings
  - Fix SQL injection vulnerabilities
  - Implement CSRF protection
  - Add input validation

- [ ] Compliance requirements
  - GDPR data handling
  - Audit logging
  - Data encryption at rest

## Infrastructure

- [ ] CI/CD pipeline
  - Automated testing on PR
  - Staging environment deployment
  - Production deployment workflow

- [ ] Monitoring & alerting
  - Application metrics (Prometheus)
  - Error tracking (Sentry)
  - Uptime monitoring
