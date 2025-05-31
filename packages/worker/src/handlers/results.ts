import { Env } from '../types';

export async function handleResults(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const raceId = url.pathname.split('/').pop();

  try {
    const { results } = await env.DB.prepare(
      `SELECT 
        rr.*,
        d.name as driver_name,
        d.code as driver_code,
        d.team as driver_team,
        r.name as race_name,
        r.date as race_date
       FROM race_results rr
       JOIN drivers d ON rr.driver_id = d.id
       JOIN races r ON rr.race_id = r.id
       WHERE rr.race_id = ?
       ORDER BY rr.position`
    ).bind(raceId).all();

    return new Response(JSON.stringify({ results }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Failed to fetch results' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}