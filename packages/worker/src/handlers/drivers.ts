import { Env } from '../types';

export async function handleDrivers(request: Request, env: Env): Promise<Response> {
  try {
    const { results } = await env.DB.prepare(
      `SELECT 
        d.*,
        COUNT(DISTINCT p.race_id) as total_predictions,
        AVG(p.predicted_position) as avg_predicted_position,
        COUNT(DISTINCT rr.race_id) as total_races,
        AVG(rr.position) as avg_actual_position,
        SUM(rr.points) as total_points
       FROM drivers d
       LEFT JOIN predictions p ON d.id = p.driver_id
       LEFT JOIN race_results rr ON d.id = rr.driver_id
       GROUP BY d.id
       ORDER BY total_points DESC`
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