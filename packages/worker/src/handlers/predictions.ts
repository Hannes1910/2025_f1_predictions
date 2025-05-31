import { Env } from '../types';

export async function handlePredictions(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const raceId = url.pathname.split('/').pop();

  try {
    const { results } = await env.DB.prepare(
      `SELECT 
        p.*,
        d.name as driver_name,
        d.code as driver_code,
        d.team as driver_team,
        r.name as race_name,
        r.date as race_date
       FROM predictions p
       JOIN drivers d ON p.driver_id = d.id
       JOIN races r ON p.race_id = r.id
       WHERE p.race_id = ?
       ORDER BY p.predicted_position`
    ).bind(raceId).all();

    return new Response(JSON.stringify({ predictions: results }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Failed to fetch predictions' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

export async function handleLatestPredictions(request: Request, env: Env): Promise<Response> {
  try {
    // Get the latest race
    const { results: races } = await env.DB.prepare(
      `SELECT id FROM races 
       WHERE date >= date('now') 
       ORDER BY date ASC 
       LIMIT 1`
    ).all();

    if (races.length === 0) {
      // If no future races, get the most recent past race
      const { results: pastRaces } = await env.DB.prepare(
        `SELECT id FROM races 
         ORDER BY date DESC 
         LIMIT 1`
      ).all();
      
      if (pastRaces.length === 0) {
        return new Response(JSON.stringify({ predictions: [] }), {
          headers: { 'Content-Type': 'application/json' },
        });
      }
      
      races[0] = pastRaces[0];
    }

    const raceId = races[0].id;

    // Get predictions for this race
    const { results: predictions } = await env.DB.prepare(
      `SELECT 
        p.*,
        d.name as driver_name,
        d.code as driver_code,
        d.team as driver_team,
        r.name as race_name,
        r.date as race_date,
        r.circuit as race_circuit
       FROM predictions p
       JOIN drivers d ON p.driver_id = d.id
       JOIN races r ON p.race_id = r.id
       WHERE p.race_id = ?
       ORDER BY p.predicted_position`
    ).bind(raceId).all();

    return new Response(JSON.stringify({ predictions }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Failed to fetch latest predictions' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}