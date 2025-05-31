import { Env } from '../types';

export async function handleDrivers(request: Request, env: Env): Promise<Response> {
  try {
    const { results } = await env.DB.prepare(
      `SELECT 
        d.*,
        COALESCE(pred.total_predictions, 0) as total_predictions,
        pred.avg_predicted_position,
        COALESCE(res.total_races, 0) as total_races,
        res.avg_actual_position,
        COALESCE(res.total_points, 0) as total_points
       FROM drivers d
       LEFT JOIN (
         SELECT 
           driver_id,
           COUNT(DISTINCT race_id) as total_predictions,
           AVG(predicted_position) as avg_predicted_position
         FROM predictions 
         GROUP BY driver_id
       ) pred ON d.id = pred.driver_id
       LEFT JOIN (
         SELECT 
           driver_id,
           COUNT(DISTINCT race_id) as total_races,
           AVG(position) as avg_actual_position,
           SUM(points) as total_points
         FROM race_results 
         GROUP BY driver_id
       ) res ON d.id = res.driver_id
       ORDER BY COALESCE(res.total_points, 0) DESC`
    ).all();

    return new Response(JSON.stringify({ drivers: results }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Failed to fetch drivers' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}