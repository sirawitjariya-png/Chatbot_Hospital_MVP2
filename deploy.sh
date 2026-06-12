#!/usr/bin/env bash
# Build, push, and deploy to Cloud Run.
# Usage: ./deploy.sh [IMAGE_TAG]
# Requires: gcloud CLI authenticated, Docker running, Artifact Registry repo created.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT="phupha-chatbot"
GCP_REGION="asia-southeast1"           # change if you prefer another region
AR_REPO="chatbot-repo"                 # Artifact Registry repo name
SERVICE_NAME="hospital-chatbot-v3"     # Cloud Run service name
IMAGE_TAG="${1:-latest}"

IMAGE="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME:$IMAGE_TAG"
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Rebuilding vector index (data/chroma/) from data/raw/..."
python main.py ingest

echo "==> Authenticating Docker with Artifact Registry..."
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev" --quiet

echo "==> Building image: $IMAGE"
docker build --platform linux/amd64 -t "$IMAGE" .

echo "==> Pushing image..."
docker push "$IMAGE"

echo "==> Deploying to Cloud Run ($SERVICE_NAME)..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$GCP_REGION" \
  --platform managed \
  --allow-unauthenticated \
  --execution-environment gen2 \
  --cpu-boost \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 3 \
  --concurrency 10 \
  --timeout 120 \
  --port 8000 \
  --env-vars-file .env.yaml \
  --project "$GCP_PROJECT"

echo ""
echo "==> Done. Service URL:"
gcloud run services describe "$SERVICE_NAME" \
  --region "$GCP_REGION" \
  --project "$GCP_PROJECT" \
  --format "value(status.url)"
