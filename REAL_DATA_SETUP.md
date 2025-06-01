# Real Data Integration Setup

This guide explains how to set up real F1 data integration instead of using mock data.

## Overview

The system now fetches real-time data from:
- **FastF1 API**: For qualifying times and race results
- **Open-Meteo API**: For weather data (free, no API key required)

## 1. Remove Mock Data

First, clean up all mock data from your database:

```bash
# Run the cleanup script
wrangler d1 execute f1-predictions-db --file=./scripts/remove_mock_data.sql
```

## 2. Deploy FastF1 Service

The FastF1 Python library needs a bridge service since Cloudflare Workers run JavaScript.

### Option A: Deploy to Cloud Run (Recommended)

1. Create a Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fastf1_service.py .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "fastf1_service:app", "--host", "0.0.0.0", "--port", "8080"]
```

2. Deploy to Google Cloud Run:
```bash
gcloud run deploy fastf1-service \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option B: Deploy to Railway/Render

1. Connect your GitHub repo
2. Set the start command: `python fastf1_service.py`
3. Deploy

### Option C: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python fastf1_service.py
```

## 3. Configure Worker Environment

Add the FastF1 service URL to your Worker:

```bash
# Add to wrangler.toml
[vars]
FASTF1_SERVICE_URL = "https://your-fastf1-service-url.com"

# Or set as secret
wrangler secret put FASTF1_SERVICE_URL
```

## 4. Deploy Updated Worker

```bash
cd packages/worker
npm run build
npm run deploy
```

## 5. How It Works

### Qualifying Data
- When a user requests race details, the system checks if qualifying data exists
- If not and the race weekend has started (within 3 days), it fetches from FastF1
- Data is cached in the database for future requests

### Race Results
- Similar to qualifying, but only fetches if the race date has passed
- Automatically calculates championship points based on finishing positions

### Weather Data
- Uses Open-Meteo API (free, no key required)
- Fetches real weather data for race locations
- Falls back to historical patterns if API is unavailable

## 6. Data Flow

1. User requests race details → Worker checks database
2. If no data exists → Worker calls FastF1 service
3. FastF1 service fetches from official F1 data
4. Data is stored in D1 database
5. Future requests use cached data

## 7. Testing

Test the integration:

```bash
# Check if FastF1 service is running
curl https://your-fastf1-service-url.com/health

# Test qualifying data fetch
curl https://your-worker.workers.dev/api/qualifying/1

# Test race results fetch  
curl https://your-worker.workers.dev/api/race/1
```

## 8. Monitoring

- Check Worker logs for any FastF1 fetch errors
- Monitor FastF1 service health endpoint
- Set up alerts for failed data fetches

## Important Notes

- FastF1 data is typically available within hours of sessions ending
- Respect rate limits - data is cached after first fetch
- The system gracefully falls back to empty data if fetches fail
- All times are stored in seconds for consistency