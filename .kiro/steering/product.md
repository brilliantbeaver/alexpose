# AlexPose - Product Overview

AlexPose is an AI-powered gait analysis platform that processes video input to identify normal vs abnormal gait patterns and classify specific health conditions.

## Core Capabilities

- Multi-format video analysis (MP4, AVI, MOV, WebM, YouTube URLs)
- Pose estimation using MediaPipe, OpenPose, Ultralytics, or AlphaPose
- **82 comprehensive gait features** (cadence, joint angles, velocity, symmetry, stability)
- **Configurable confidence thresholds** (default: 0.3) for robust real-world video analysis
- LLM-based classification (OpenAI GPT, Google Gemini) with two-stage analysis
- Clinical condition identification (antalgic, hemiplegic, parkinsonian, etc.)

## Analysis Pipeline

Video Input → Pose Estimation → Feature Extraction → Temporal Analysis → Symmetry Analysis → LLM Classification → Clinical Report

## Key Modules

- `ambient/` - Core Python library for gait analysis
- `server/` - FastAPI backend API
- `frontend/` - Next.js web interface
- `config/` - YAML configuration files

## Data Sources

- GAVD (Gait Abnormality Video Dataset) for training/validation
- User-uploaded videos
- YouTube video URLs
