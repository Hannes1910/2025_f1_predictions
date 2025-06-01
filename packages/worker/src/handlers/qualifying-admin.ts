/**
 * Qualifying Admin Handler
 * Endpoints for managing qualifying results
 */

import { Env } from '../types';

/**
 * Add or update qualifying results for a race
 */
export async function handleCreateQualifyingResults(request: Request, env: Env): Promise<Response> {
  try {
    // Check for API key authentication
    const apiKey = request.headers.get('X-API-Key');
    if (apiKey !== env.PREDICTIONS_API_KEY) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Parse request body
    const data = await request.json() as {
      race_id: number;
      results: Array<{
        driver_id: number;
        q1_time?: number;
        q2_time?: number;
        q3_time?: number;
        qualifying_time: number;
        grid_position: number;
        qualifying_position?: number;
        grid_penalty?: number;
      }>;
    };

    // Validate required fields
    if (!data.race_id || !data.results || !Array.isArray(data.results)) {
      return new Response(JSON.stringify({ error: 'Missing required fields' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const timestamp = new Date().toISOString();

    // Delete existing qualifying results for this race
    await env.DB.prepare('DELETE FROM qualifying_results WHERE race_id = ?')
      .bind(data.race_id)
      .run();

    // Insert new qualifying results
    const insertPromises = data.results.map(result => 
      env.DB.prepare(`
        INSERT INTO qualifying_results 
        (race_id, driver_id, q1_time, q2_time, q3_time, qualifying_time, 
         grid_position, qualifying_position, grid_penalty, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(
        data.race_id,
        result.driver_id,
        result.q1_time || null,
        result.q2_time || null,
        result.q3_time || null,
        result.qualifying_time,
        result.grid_position,
        result.qualifying_position || result.grid_position,
        result.grid_penalty || 0,
        timestamp
      ).run()
    );

    await Promise.all(insertPromises);

    return new Response(JSON.stringify({
      success: true,
      race_id: data.race_id,
      results_stored: data.results.length,
      created_at: timestamp
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error creating qualifying results:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to store qualifying results',
      details: error instanceof Error ? error.message : String(error)
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get qualifying results for a specific race
 */
export async function handleGetQualifyingResults(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const raceId = url.pathname.split('/').pop();

    if (!raceId) {
      return new Response(JSON.stringify({ error: 'Race ID required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get qualifying results with driver info
    const { results } = await env.DB.prepare(`
      SELECT 
        qr.*,
        d.code as driver_code,
        d.name as driver_name,
        d.team as driver_team
      FROM qualifying_results qr
      JOIN drivers d ON qr.driver_id = d.id
      WHERE qr.race_id = ?
      ORDER BY qr.grid_position
    `).bind(raceId).all();

    // Get race info
    const race = await env.DB.prepare(`
      SELECT * FROM races WHERE id = ?
    `).bind(raceId).first();

    return new Response(JSON.stringify({
      race,
      qualifying_results: results,
      count: results.length
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching qualifying results:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to fetch qualifying results' 
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}