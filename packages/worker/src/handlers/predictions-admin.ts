import { Env } from '../types'

interface PredictionRequest {
  race_id: number
  predictions: Array<{
    driver_id: number
    predicted_position: number
    predicted_time: number
    confidence: number
  }>
  model_version: string
  model_metrics?: {
    mae: number
    accuracy: number
  }
}

export async function handleCreatePredictions(request: Request, env: Env): Promise<Response> {
  try {
    // Check for API key authentication
    const apiKey = request.headers.get('X-API-Key')
    if (apiKey !== env.PREDICTIONS_API_KEY) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // Parse request body
    const data: PredictionRequest = await request.json()

    // Validate required fields
    if (!data.race_id || !data.predictions || !data.model_version) {
      return new Response(JSON.stringify({ error: 'Missing required fields' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // Start a transaction
    const timestamp = new Date().toISOString()
    
    // Delete existing predictions for this race
    await env.DB.prepare(
      'DELETE FROM predictions WHERE race_id = ?'
    ).bind(data.race_id).run()

    // Insert new predictions
    const insertPromises = data.predictions.map(pred =>
      env.DB.prepare(
        `INSERT INTO predictions 
         (race_id, driver_id, predicted_position, predicted_time, confidence, model_version, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        data.race_id,
        pred.driver_id,
        pred.predicted_position,
        pred.predicted_time,
        pred.confidence,
        data.model_version,
        timestamp
      ).run()
    )

    await Promise.all(insertPromises)

    // Store model metrics if provided
    if (data.model_metrics) {
      await env.DB.prepare(
        `INSERT INTO model_metrics (model_version, race_id, mae, accuracy, created_at)
         VALUES (?, ?, ?, ?, ?)`
      ).bind(
        data.model_version,
        data.race_id,
        data.model_metrics.mae,
        data.model_metrics.accuracy,
        timestamp
      ).run()
    }

    return new Response(JSON.stringify({ 
      success: true, 
      predictions_stored: data.predictions.length,
      model_version: data.model_version
    }), {
      headers: { 'Content-Type': 'application/json' },
    })

  } catch (error) {
    console.error('Error creating predictions:', error)
    return new Response(JSON.stringify({ error: 'Failed to store predictions' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}

export async function handleTriggerPredictions(request: Request, env: Env): Promise<Response> {
  try {
    // Check for API key or cron trigger
    const apiKey = request.headers.get('X-API-Key')
    const isCronTrigger = request.headers.get('X-CF-Cron-Trigger') === 'true'
    
    if (apiKey !== env.PREDICTIONS_API_KEY && !isCronTrigger) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // Get upcoming races
    const { results: upcomingRaces } = await env.DB.prepare(
      `SELECT r.*, COUNT(p.id) as prediction_count
       FROM races r
       LEFT JOIN predictions p ON r.id = p.race_id
       WHERE r.date >= date('now')
         AND r.date <= date('now', '+14 days')
         AND r.season = 2025
       GROUP BY r.id
       ORDER BY r.date
       LIMIT 5`
    ).all()

    const racesToProcess = upcomingRaces.filter(race => race.prediction_count === 0)

    if (racesToProcess.length === 0) {
      return new Response(JSON.stringify({ 
        message: 'No races need predictions',
        checked: upcomingRaces.length
      }), {
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // Trigger prediction generation via external service
    // In production, this would call your Python service or GitHub Action
    const results = []
    for (const race of racesToProcess) {
      // Log the race that needs predictions
      results.push({
        race_id: race.id,
        race_name: race.name,
        date: race.date,
        status: 'pending'
      })
    }

    // Store trigger event
    await env.DB.prepare(
      `INSERT INTO prediction_triggers (races_count, trigger_type, created_at)
       VALUES (?, ?, ?)`
    ).bind(
      racesToProcess.length,
      isCronTrigger ? 'cron' : 'manual',
      new Date().toISOString()
    ).run()

    return new Response(JSON.stringify({ 
      success: true,
      races_to_process: results,
      message: `Triggered predictions for ${racesToProcess.length} races`
    }), {
      headers: { 'Content-Type': 'application/json' },
    })

  } catch (error) {
    console.error('Error triggering predictions:', error)
    return new Response(JSON.stringify({ error: 'Failed to trigger predictions' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}