import { Env } from '../types';

export async function handleAccuracy(request: Request, env: Env): Promise<Response> {
  try {
    // Get model accuracy over time
    const { results: metrics } = await env.DB.prepare(
      `SELECT 
        mm.*,
        r.name as race_name,
        r.date as race_date
       FROM model_metrics mm
       LEFT JOIN races r ON mm.race_id = r.id
       ORDER BY mm.created_at DESC
       LIMIT 20`
    ).all();

    // Get prediction accuracy by comparing predictions to actual results
    const { results: accuracy } = await env.DB.prepare(
      `SELECT 
        r.name as race_name,
        r.date as race_date,
        AVG(ABS(p.predicted_position - rr.position)) as avg_position_error,
        AVG(ABS(p.predicted_time - rr.time)) as avg_time_error,
        COUNT(*) as prediction_count
       FROM predictions p
       JOIN race_results rr ON p.race_id = rr.race_id AND p.driver_id = rr.driver_id
       JOIN races r ON p.race_id = r.id
       GROUP BY p.race_id
       ORDER BY r.date DESC
       LIMIT 10`
    ).all();

    return new Response(JSON.stringify({ 
      model_metrics: metrics,
      prediction_accuracy: accuracy 
    }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Failed to fetch analytics' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}