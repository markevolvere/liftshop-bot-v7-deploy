#!/usr/bin/env bash
# =============================================================================
# Lift Shop Bot v7 — Full Deployment Script
# Run this in Azure Cloud Shell (bash)
# =============================================================================
set -euo pipefail

RG="MjeanesResourceGroup"
APP="liftshop-teams-bot"
ACR="liftshopbotacr"
IMAGE="liftshop-teams-bot:latest"

echo "============================================================"
echo " Lift Shop Bot v7 Deployment"
echo "============================================================"

# ── Step 0: Clone this repo ───────────────────────────────────────────────────
REPO_DIR="$HOME/bot-v7-patch"
if [ -d "$REPO_DIR" ]; then
    echo "Updating existing clone..."
    git -C "$REPO_DIR" pull
else
    echo "Cloning deployment repo..."
    git clone https://github.com/markevolvere/liftshop-bot-v7-deploy.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── Step 1: Get secrets from Azure ───────────────────────────────────────────
echo ""
echo "[1/5] Fetching environment variables from Azure..."
export DATABASE_URL=$(az webapp config appsettings list \
    --name "$APP" --resource-group "$RG" \
    --query "[?name=='DATABASE_URL'].value" -o tsv)
export VOYAGE_API_KEY=$(az webapp config appsettings list \
    --name "$APP" --resource-group "$RG" \
    --query "[?name=='VOYAGE_API_KEY'].value" -o tsv)

if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL not found. Check App Service settings."
    exit 1
fi
if [ -z "$VOYAGE_API_KEY" ]; then
    echo "ERROR: VOYAGE_API_KEY not found. Check App Service settings."
    exit 1
fi
echo "  DATABASE_URL: ${DATABASE_URL:0:40}..."
echo "  VOYAGE_API_KEY: ${VOYAGE_API_KEY:0:10}..."

# ── Step 2: Apply SQL files ───────────────────────────────────────────────────
echo ""
echo "[2/5] Applying SQL files to PostgreSQL..."

# Install psql if needed
if ! command -v psql &>/dev/null; then
    echo "  Installing postgresql-client..."
    sudo apt-get install -y postgresql-client -q
fi

# Apply chunks SQL
if [ -f "ocr_documents_chunks.sql" ]; then
    echo "  Applying ocr_documents_chunks.sql (this may take 1-2 min)..."
    psql "$DATABASE_URL" -f ocr_documents_chunks.sql -v ON_ERROR_STOP=0 --quiet
    echo "  ✅ ocr_documents_chunks.sql applied"
else
    echo "  ⚠️  ocr_documents_chunks.sql not found in repo."
    echo "  Upload it to Cloud Shell manually, then run:"
    echo "    psql \"\$DATABASE_URL\" -f ~/ocr_documents_chunks.sql"
fi

# Apply facts SQL
if [ -f "ocr_facts.sql" ]; then
    echo "  Applying ocr_facts.sql (this may take 1-2 min)..."
    psql "$DATABASE_URL" -f ocr_facts.sql -v ON_ERROR_STOP=0 --quiet
    echo "  ✅ ocr_facts.sql applied"
else
    echo "  ⚠️  ocr_facts.sql not found in repo."
    echo "  Upload it to Cloud Shell manually, then run:"
    echo "    psql \"\$DATABASE_URL\" -f ~/ocr_facts.sql"
fi

# ── Step 3: Run embeddings ────────────────────────────────────────────────────
echo ""
echo "[3/5] Running Voyage AI embeddings for NULL rows..."
pip install voyageai psycopg2-binary --user -q
python3 embed_new_chunks.py
echo "  ✅ Embeddings complete"

# ── Step 4: Build Docker image ────────────────────────────────────────────────
echo ""
echo "[4/5] Building and pushing Docker image v7 via ACR..."
az acr build \
    --registry "$ACR" \
    --image "$IMAGE" \
    -f Dockerfile.patch \
    . \
    --no-logs
echo "  ✅ Docker image built and pushed to $ACR"

# ── Step 5: Restart App Service ───────────────────────────────────────────────
echo ""
echo "[5/5] Restarting App Service..."
az webapp restart --name "$APP" --resource-group "$RG"
echo "  ✅ App Service restarted"

echo ""
echo "============================================================"
echo " v7 Deployment complete!"
echo " Scope qualifier, META model, and ${DATABASE_URL:0:10}... chunks loaded."
echo "============================================================"
