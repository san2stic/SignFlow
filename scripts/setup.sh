#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required"
  exit 1
fi

cp -n .env.example .env || true
docker compose build

echo "Setup complete. Run: docker compose up"
