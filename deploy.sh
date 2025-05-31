#!/bin/bash

echo "🚀 Starting F1 Predictions deployment to Cloudflare..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Deploy Worker
echo -e "${BLUE}📦 Building and deploying Worker...${NC}"
cd packages/worker
npm run build
npx wrangler deploy

# Get the Worker URL
echo -e "${GREEN}✅ Worker deployed!${NC}"
echo "Worker URL will be: https://f1-predictions-api.YOUR-SUBDOMAIN.workers.dev"

# Build Frontend
echo -e "${BLUE}📦 Building frontend...${NC}"
cd ../frontend
npm run build

# Deploy to Cloudflare Pages
echo -e "${BLUE}☁️  Deploying frontend to Cloudflare Pages...${NC}"
npx wrangler pages deploy dist --project-name=f1-predictions-dashboard

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "Remember to:"
echo "1. Update the VITE_API_URL in .env.production with your actual Worker URL"
echo "2. Rebuild and redeploy frontend after updating the URL"
echo "3. Set up custom domains if desired"