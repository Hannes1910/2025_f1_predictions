import { Router } from 'itty-router';
import { Env } from './types';
import { handlePredictions, handleLatestPredictions } from './handlers/predictions';
import { handleResults } from './handlers/results';
import { handleDrivers } from './handlers/drivers';
import { handleRaces } from './handlers/races';
import { handleAccuracy } from './handlers/analytics';

const router = Router();

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

// Handle CORS preflight
router.options('*', () => {
  return new Response(null, { headers: corsHeaders });
});

// API Routes
router.get('/api/predictions/latest', handleLatestPredictions);
router.get('/api/predictions/:raceId', handlePredictions);
router.get('/api/results/:raceId', handleResults);
router.get('/api/drivers', handleDrivers);
router.get('/api/races', handleRaces);
router.get('/api/analytics/accuracy', handleAccuracy);

// Health check
router.get('/api/health', () => {
  return new Response(JSON.stringify({ status: 'ok' }), {
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
});

// 404 handler
router.all('*', () => {
  return new Response('Not Found', { status: 404 });
});

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return router.handle(request, env).then((response: Response) => {
      // Add CORS headers to all responses
      Object.entries(corsHeaders).forEach(([key, value]) => {
        response.headers.set(key, value);
      });
      return response;
    });
  },
};