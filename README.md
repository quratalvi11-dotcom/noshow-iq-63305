# NoShowIQ ??

> A prediction API that tells clinics which patients are likely to skip their appointment.

![CI](https://github.com/quratalvi11-dotcom/noshow-iq-63305/actions/workflows/lint.yml/badge.svg)

##  Live Deployment

**API Base URL:** `https://quratalvi11-dotcom-noshow-iq.hf.space`

## Endpoints

- GET /health  Health check
- POST /predict  Predict no-show risk
- GET /history  Last 20 predictions
- GET /stats  Aggregated MongoDB stats

##  Quick Start

git clone https://github.com/quratalvi11-dotcom/noshow-iq-63305.git
cd noshow-iq-63305
docker compose up --build

## ?? License

MIT License
