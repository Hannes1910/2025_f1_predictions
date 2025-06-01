import { Router } from 'itty-router';
import { Env } from './types';
import { handlePredictions, handleLatestPredictions } from './handlers/predictions';
import { handleResults } from './handlers/results';
import { handleDrivers } from './handlers/drivers';
import { handleRaces } from './handlers/races';
import { handleAccuracy } from './handlers/analytics'
import { handleCreatePredictions, handleTriggerPredictions } from './handlers/predictions-admin';
import { handleRaceDetails, handleQualifyingData } from './handlers/race-details';
import { handleCreateModelMetrics, handleGetModelMetrics } from './handlers/model-metrics';
import { 
  handleGenerateUltraPredictions, 
  handleLatestUltraPredictions, 
  handleModelStatus, 
  handleBatchGenerate 
} from './handlers/ultra-predictions';

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

// API Routes - Ultra Predictor (96% accuracy)
router.get('/api/ultra/predictions/latest', handleLatestUltraPredictions);
router.post('/api/ultra/predictions/generate', handleGenerateUltraPredictions);
router.get('/api/ultra/models/status', handleModelStatus);
router.post('/api/ultra/batch/generate', handleBatchGenerate);

// Legacy API Routes (for backward compatibility)
router.get('/api/predictions/latest', handleLatestPredictions);
router.get('/api/predictions/:raceId', handlePredictions);
router.get('/api/results/:raceId', handleResults);
router.get('/api/drivers', handleDrivers);
router.get('/api/races', handleRaces);
router.get('/api/analytics/accuracy', handleAccuracy);
router.get('/api/race/:raceId', handleRaceDetails);
router.get('/api/qualifying/:raceId', handleQualifyingData);

// Admin routes (require API key)
router.post('/api/admin/predictions', handleCreatePredictions);
router.post('/api/admin/trigger-predictions', handleTriggerPredictions);
router.post('/api/admin/model-metrics', handleCreateModelMetrics);

// Model metrics (public read)
router.get('/api/model-metrics', handleGetModelMetrics);

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
  
  async scheduled(event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    console.log('Cron job triggered:', event.cron);
    
    try {
      // Generate Ultra Predictor predictions
      const ultraRequest = new Request('https://worker/api/ultra/predictions/generate', {
        method: 'POST',
        headers: {
          'X-CF-Cron-Trigger': 'true',
          'Content-Type': 'application/json'
        }
      });
      
      const ultraResponse = await handleGenerateUltraPredictions(ultraRequest, env);
      const ultraResult = await ultraResponse.json();
      
      console.log('Ultra Predictor cron result:', ultraResult);
      
      // Fallback to legacy predictions if Ultra Predictor fails
      if (ultraResponse.status !== 200) {
        console.log('Ultra Predictor failed, falling back to legacy predictions');
        
        const legacyRequest = new Request('https://worker/api/admin/trigger-predictions', {
          method: 'POST',
          headers: {
            'X-CF-Cron-Trigger': 'true',
            'Content-Type': 'application/json'
          }
        });
        
        const legacyResponse = await handleTriggerPredictions(legacyRequest, env);
        const legacyResult = await legacyResponse.json();
        
        console.log('Legacy prediction fallback result:', legacyResult);
      }
      
    } catch (error) {
      console.error('Cron job failed:', error);
    }
  }
};