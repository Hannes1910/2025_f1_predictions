/**
 * Data API Handler
 * Provides database access for ML service since D1 is edge-only
 */

import { Env } from '../types';

/**
 * Get driver performance statistics
 */
export async function handleGetDriverStats(request: Request, env: Env): Promise<Response> {
  try {
    // Extract driver ID from URL
    const url = new URL(request.url);
    const driverId = url.pathname.split('/').pop();
    
    if (!driverId) {
      return new Response(JSON.stringify({ error: 'Driver ID required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get driver statistics from D1
    const stats = await env.DB.prepare(`
      SELECT 
        d.id,
        d.code,
        d.name,
        d.team,
        d.points as championship_points,
        (SELECT AVG(CAST(position AS REAL)) 
         FROM race_results 
         WHERE driver_id = d.id 
         ORDER BY race_id DESC 
         LIMIT 3) as recent_form,
        (SELECT AVG(CAST(position AS REAL)) 
         FROM race_results 
         WHERE driver_id = d.id) as avg_finish_position,
        (SELECT COUNT(*) * 1.0 / NULLIF((SELECT COUNT(*) FROM race_results WHERE driver_id = d.id), 0)
         FROM race_results 
         WHERE driver_id = d.id AND status != 'Finished') as dnf_rate,
        (SELECT AVG(grid_position) 
         FROM qualifying_results 
         WHERE driver_id = d.id) as avg_grid_position
      FROM drivers d
      WHERE d.id = ?
    `).bind(driverId).first();

    if (!stats) {
      return new Response(JSON.stringify({ error: 'Driver not found' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify(stats), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching driver stats:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to fetch driver statistics',
      details: error instanceof Error ? error.message : String(error)
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get team performance data
 */
export async function handleGetTeamStats(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const team = decodeURIComponent(url.pathname.split('/').pop() || '');

    const stats = await env.DB.prepare(`
      SELECT 
        team,
        AVG(CAST(rr.position AS REAL)) as avg_position,
        COUNT(DISTINCT rr.race_id) as races_participated,
        SUM(rr.points) as total_points,
        COUNT(DISTINCT d.id) as driver_count
      FROM drivers d
      JOIN race_results rr ON d.id = rr.driver_id
      WHERE d.team = ?
      GROUP BY d.team
    `).bind(team).first();

    return new Response(JSON.stringify(stats || {}), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching team stats:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to fetch team statistics' 
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get race features for ML prediction
 */
export async function handleGetRaceFeatures(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const raceId = url.pathname.split('/').pop();

    if (!raceId) {
      return new Response(JSON.stringify({ error: 'Race ID required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get race info
    const race = await env.DB.prepare(`
      SELECT * FROM races WHERE id = ?
    `).bind(raceId).first();

    if (!race) {
      return new Response(JSON.stringify({ error: 'Race not found' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get all drivers with their features
    const { results: driverFeatures } = await env.DB.prepare(`
      SELECT 
        d.*,
        -- Recent form (last 3 races)
        (SELECT AVG(CAST(position AS REAL)) 
         FROM race_results 
         WHERE driver_id = d.id 
         ORDER BY race_id DESC 
         LIMIT 3) as recent_form,
        -- Average finish
        (SELECT AVG(CAST(position AS REAL)) 
         FROM race_results 
         WHERE driver_id = d.id) as avg_finish_position,
        -- DNF rate
        (SELECT COUNT(*) * 1.0 / NULLIF((SELECT COUNT(*) FROM race_results WHERE driver_id = d.id), 0)
         FROM race_results 
         WHERE driver_id = d.id AND status != 'Finished') as dnf_rate,
        -- Team average
        (SELECT AVG(CAST(rr.position AS REAL))
         FROM race_results rr
         JOIN drivers d2 ON rr.driver_id = d2.id
         WHERE d2.team = d.team) as team_performance,
        -- Qualifying for this race (if available)
        (SELECT grid_position 
         FROM qualifying_results 
         WHERE driver_id = d.id AND race_id = ?) as grid_position,
        (SELECT qualifying_time 
         FROM qualifying_results 
         WHERE driver_id = d.id AND race_id = ?) as qualifying_time
      FROM drivers d
      ORDER BY d.id
    `).bind(raceId, raceId).all();

    return new Response(JSON.stringify({
      race,
      drivers: driverFeatures
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching race features:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to fetch race features' 
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get historical patterns for a circuit
 */
export async function handleGetCircuitPatterns(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const circuit = decodeURIComponent(url.pathname.split('/').pop() || '');

    const patterns = await env.DB.prepare(`
      SELECT 
        r.circuit,
        COUNT(DISTINCT r.id) as race_count,
        AVG(CAST(rr.position AS REAL)) as avg_positions,
        COUNT(CASE WHEN rr.status != 'Finished' THEN 1 END) * 1.0 / COUNT(*) as dnf_rate,
        -- Add more circuit-specific patterns as needed
        COUNT(DISTINCT r.season) as seasons_held
      FROM races r
      JOIN race_results rr ON r.id = rr.race_id
      WHERE r.circuit = ?
      GROUP BY r.circuit
    `).bind(circuit).first();

    return new Response(JSON.stringify(patterns || {
      circuit,
      race_count: 0,
      dnf_rate: 0.1,
      message: 'No historical data for this circuit'
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching circuit patterns:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to fetch circuit patterns' 
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Batch endpoint to get all data needed for ML prediction
 */
export async function handleGetMLPredictionData(request: Request, env: Env): Promise<Response> {
  try {
    const { raceId } = await request.json() as { raceId: number };

    // Get all data in parallel
    const [race, drivers, recentResults] = await Promise.all([
      // Race info
      env.DB.prepare('SELECT * FROM races WHERE id = ?').bind(raceId).first(),
      
      // All drivers with current stats
      env.DB.prepare(`
        SELECT d.*, 
          (SELECT AVG(CAST(position AS REAL)) FROM race_results WHERE driver_id = d.id ORDER BY race_id DESC LIMIT 3) as recent_form,
          (SELECT AVG(CAST(position AS REAL)) FROM race_results WHERE driver_id = d.id) as avg_finish
        FROM drivers d
      `).all(),
      
      // Recent race results for context
      env.DB.prepare(`
        SELECT * FROM race_results 
        WHERE race_id < ? 
        ORDER BY race_id DESC 
        LIMIT 60
      `).bind(raceId).all()
    ]);

    return new Response(JSON.stringify({
      race,
      drivers: drivers.results,
      recentResults: recentResults.results,
      timestamp: new Date().toISOString()
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching ML prediction data:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to fetch ML prediction data' 
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}