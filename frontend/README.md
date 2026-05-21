# Frontend Web Interface

This directory contains the web-based frontend for visualizing the autonomous driving system's telemetry and status.

## Current Status
The frontend is a React + TypeScript application that connects to the backend's WebSocket telemetry at `/ws/telemetry`. It provides a real-time dashboard for monitoring agent performance.

## Structure
- `web/`: Contains the web application source and Dockerfile.

## Running with Docker
The frontend can be served using the provided Dockerfile, which currently serves a static placeholder.
