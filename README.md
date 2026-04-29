---
title: NoShowIQ
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# NoShowIQ 🏥

[![CI/CD](https://github.com/quratalvi11-dotcom/noshow-iq-63305/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/quratalvi11-dotcom/noshow-iq-63305/actions/workflows/ci-cd.yml)

A production-ready MLOps project that predicts whether a patient will miss their medical appointment. Built with FastAPI, MongoDB, Docker, and deployed on Hugging Face Spaces.

**Live URL:** https://Qurat-09-noshow-iq.hf.space

## Problem
Every day, clinics lose revenue because patients book appointments and never show up. NoShowIQ predicts which patients are likely to skip so clinics can act in advance.

## Dataset
- 110,000+ real appointment records from Brazilian clinics
- Source: Kaggle Medical Appointment No-Shows
- Class imbalance handled: 80% show, 20% no-show

## Tech Stack
- **API:** FastAPI + Uvicorn
- **ML:** Scikit-learn (Logistic Regression with class balancing)
- **Database:** MongoDB Atlas
- **Containerization:** Docker + Docker Compose
- **Deployment:** Hugging Face Spaces
- **CI/CD:** GitHub Actions

## API Endpoints
- `GET /health` — Health check
- `POST /predict` — Predict no-show risk for one appointment
- `GET /history` — Last 20 predictions
- `GET /stats` — Aggregated prediction statistics

## Docker Hub
https://hub.docker.com/r/qurat38027/noshow-iq
