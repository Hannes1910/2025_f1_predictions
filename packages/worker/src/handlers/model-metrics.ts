import { Env } from '../types'

export async function handleCreateModelMetrics(request: Request, env: Env): Promise<Response> {
  try {
    // Check API key
    const apiKey = request.headers.get('X-API-Key')
    if (apiKey !== env.PREDICTIONS_API_KEY) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    const data = await request.json() as any

    // Validate required fields
    if (!data.model_version || data.mae === undefined || data.accuracy === undefined) {
      return new Response(JSON.stringify({ error: 'Missing required fields' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // Insert model metrics
    const result = await env.DB.prepare(`
      INSERT OR REPLACE INTO model_metrics (model_version, race_id, mae, accuracy, created_at)
      VALUES (?, ?, ?, ?, ?)
    `).bind(
      data.model_version,
      data.race_id || null,
      data.mae,
      data.accuracy,
      data.created_at || new Date().toISOString()
    ).run()

    return new Response(JSON.stringify({ 
      success: true,
      model_version: data.model_version,
      mae: data.mae,
      accuracy: data.accuracy
    }), {
      headers: { 'Content-Type': 'application/json' },
    })

  } catch (error) {
    console.error('Error creating model metrics:', error)
    return new Response(JSON.stringify({ error: 'Failed to create model metrics' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}

export async function handleGetModelMetrics(request: Request, env: Env): Promise<Response> {
  try {
    const { results } = await env.DB.prepare(`
      SELECT mm.*, r.name as race_name, r.circuit, r.date as race_date
      FROM model_metrics mm
      LEFT JOIN races r ON mm.race_id = r.id
      ORDER BY mm.created_at DESC
      LIMIT 100
    `).all()

    // Group by model version for evolution tracking
    const modelEvolution: Record<string, any[]> = {}
    
    for (const metric of results) {
      const version = metric.model_version as string
      if (!modelEvolution[version]) {
        modelEvolution[version] = []
      }
      modelEvolution[version].push(metric)
    }

    // Calculate overall stats
    const overallStats = {
      total_models: Object.keys(modelEvolution).length,
      total_predictions: results.length,
      best_mae: Math.min(...results.map(r => Number(r.mae))),
      best_accuracy: Math.max(...results.map(r => Number(r.accuracy))),
      latest_model: results[0]?.model_version || 'none'
    }

    return new Response(JSON.stringify({ 
      metrics: results,
      evolution: modelEvolution,
      stats: overallStats
    }), {
      headers: { 'Content-Type': 'application/json' },
    })

  } catch (error) {
    console.error('Error fetching model metrics:', error)
    return new Response(JSON.stringify({ error: 'Failed to fetch model metrics' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}