/**
 * Clean Data API Handler - PRODUCTION VERSION
 * - No mock data fallbacks
 * - Proper error handling
 * - Real data only
 * - Clear validation
 */

import { Env } from '../types';

interface DriverStats {
  id: number;
  code: string;
  forename: string;
  surname: string;
  team_name: string;
  total_points: number;
  avg_recent_position: number | null;
  avg_grid_position: number | null;
  races_completed: number;
  dnf_rate: number;
}

interface QualifyingResult {
  id: number;
  code: string;
  forename: string;
  surname: string;
  team_name: string;
  q1_time_ms: number | null;
  q2_time_ms: number | null;
  q3_time_ms: number | null;
  best_time_ms: number;
  qualifying_position: number;
  grid_position: number;
  grid_penalty: number;
}

interface RaceFeatures {
  race: {
    id: number;
    season: number;
    round: number;
    name: string;
    date: string;
    circuit: string;
    country: string;
  };
  qualifying_results: QualifyingResult[];
  driver_stats: DriverStats[];
}

/**
 * Get driver performance statistics - REAL DATA ONLY
 */
export async function handleGetDriverStatsClean(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const driverId = url.pathname.split('/').pop();
    
    if (!driverId || isNaN(Number(driverId))) {
      return new Response(JSON.stringify({ 
        error: 'Valid driver ID required',
        code: 'INVALID_DRIVER_ID'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get real driver statistics
    const stats = await env.DB.prepare(`
      SELECT 
        d.id,
        d.code,
        d.forename,
        d.surname,
        te.team_name,
        -- Championship points (sum of all race results)
        COALESCE((
          SELECT SUM(points) 
          FROM race_results rr 
          JOIN races r ON rr.race_id = r.id 
          WHERE rr.driver_id = d.id 
          AND r.season = (SELECT MAX(season) FROM races)
        ), 0) as total_points,
        -- Recent form (last 3 races average position)
        (
          SELECT AVG(
            CASE 
              WHEN rr.position IS NOT NULL THEN rr.position
              ELSE 21  -- DNF penalty
            END
          ) 
          FROM race_results rr
          JOIN races r ON rr.race_id = r.id
          WHERE rr.driver_id = d.id
          ORDER BY r.date DESC
          LIMIT 3
        ) as avg_recent_position,
        -- Average grid position
        (
          SELECT AVG(CAST(grid_position AS REAL))
          FROM qualifying_results
          WHERE driver_id = d.id
        ) as avg_grid_position,
        -- Races completed this season
        (
          SELECT COUNT(*)
          FROM race_results rr
          JOIN races r ON rr.race_id = r.id
          WHERE rr.driver_id = d.id
          AND r.season = (SELECT MAX(season) FROM races)
        ) as races_completed,
        -- DNF rate (position > 20 means DNF for now)
        (
          SELECT 
            CASE 
              WHEN COUNT(*) > 0 THEN 
                COUNT(CASE WHEN position > 20 THEN 1 END) * 1.0 / COUNT(*)
              ELSE 0
            END
          FROM race_results
          WHERE driver_id = d.id
        ) as dnf_rate
      FROM drivers d
      LEFT JOIN team_entries te ON d.id = te.driver_id 
        AND te.season = (SELECT MAX(season) FROM races)
      WHERE d.id = ?
    `).bind(driverId).first();

    if (!stats) {
      return new Response(JSON.stringify({ 
        error: `Driver with ID ${driverId} not found`,
        code: 'DRIVER_NOT_FOUND'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      data: stats,
      meta: {
        data_source: 'real_f1_data',
        last_updated: new Date().toISOString()
      }
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching driver stats:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      code: 'DATABASE_ERROR',
      details: error instanceof Error ? error.message : String(error)
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get team performance statistics - REAL DATA ONLY
 */
export async function handleGetTeamStatsClean(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const teamName = decodeURIComponent(url.pathname.split('/').pop() || '');

    if (!teamName) {
      return new Response(JSON.stringify({ 
        error: 'Team name required',
        code: 'MISSING_TEAM_NAME'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const stats = await env.DB.prepare(`
      SELECT 
        te.team_name,
        COUNT(DISTINCT te.driver_id) as driver_count,
        COALESCE(SUM(rr.points), 0) as total_points,
        COUNT(DISTINCT rr.race_id) as races_participated,
        AVG(
          CASE 
            WHEN rr.position IS NOT NULL THEN rr.position
            ELSE 21
          END
        ) as avg_position
      FROM team_entries te
      LEFT JOIN race_results rr ON te.driver_id = rr.driver_id
      LEFT JOIN races r ON rr.race_id = r.id AND r.season = te.season
      WHERE te.team_name = ?
      AND te.season = (SELECT MAX(season) FROM races)
      GROUP BY te.team_name
    `).bind(teamName).first();

    if (!stats) {
      return new Response(JSON.stringify({ 
        error: `Team '${teamName}' not found or no data available`,
        code: 'TEAM_NOT_FOUND'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      data: stats,
      meta: {
        data_source: 'real_f1_data',
        last_updated: new Date().toISOString()
      }
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching team stats:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      code: 'DATABASE_ERROR'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get race features for ML prediction - REAL DATA ONLY
 */
export async function handleGetRaceFeaturesClean(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const pathParts = url.pathname.split('/');
    const raceId = pathParts[pathParts.length - 2]; // Get race ID before "features"

    if (!raceId || isNaN(Number(raceId))) {
      return new Response(JSON.stringify({ 
        error: 'Valid race ID required',
        code: 'INVALID_RACE_ID'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get race info
    const race = await env.DB.prepare(`
      SELECT * FROM races WHERE id = ?
    `).bind(raceId).first();

    if (!race) {
      return new Response(JSON.stringify({ 
        error: `Race with ID ${raceId} not found`,
        code: 'RACE_NOT_FOUND'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get qualifying results for this race
    const { results: qualifyingResults } = await env.DB.prepare(`
      SELECT 
        qr.*,
        d.code,
        d.forename,
        d.surname,
        te.team_name
      FROM qualifying_results qr
      JOIN drivers d ON qr.driver_id = d.id
      LEFT JOIN team_entries te ON d.id = te.driver_id 
        AND te.season = (SELECT season FROM races WHERE id = ?)
      WHERE qr.race_id = ?
      ORDER BY qr.grid_position
    `).bind(raceId, raceId).all();

    // Get driver statistics
    const { results: driverStats } = await env.DB.prepare(`
      SELECT 
        d.id,
        d.code,
        d.forename,
        d.surname,
        te.team_name,
        -- Points this season
        COALESCE((
          SELECT SUM(points) 
          FROM race_results rr 
          JOIN races r ON rr.race_id = r.id 
          WHERE rr.driver_id = d.id 
          AND r.season = (SELECT season FROM races WHERE id = ?)
          AND r.id < ?
        ), 0) as total_points,
        -- Recent form
        (
          SELECT AVG(
            CASE 
              WHEN rr.position IS NOT NULL THEN rr.position
              ELSE 21
            END
          ) 
          FROM race_results rr
          JOIN races r ON rr.race_id = r.id
          WHERE rr.driver_id = d.id
          AND r.id < ?
          ORDER BY r.date DESC
          LIMIT 3
        ) as avg_recent_position,
        -- Average grid position
        (
          SELECT AVG(CAST(grid_position AS REAL))
          FROM qualifying_results
          WHERE driver_id = d.id
          AND race_id < ?
        ) as avg_grid_position,
        -- DNF rate
        (
          SELECT 
            CASE 
              WHEN COUNT(*) > 0 THEN 
                COUNT(CASE WHEN status != 'Finished' THEN 1 END) * 1.0 / COUNT(*)
              ELSE 0
            END
          FROM race_results
          WHERE driver_id = d.id
        ) as dnf_rate
      FROM drivers d
      LEFT JOIN team_entries te ON d.id = te.driver_id 
        AND te.season = (SELECT season FROM races WHERE id = ?)
      ORDER BY d.id
    `).bind(raceId, raceId, raceId, raceId, raceId).all();

    const response: RaceFeatures = {
      race: {
        id: race.id as number,
        season: race.season as number,
        round: race.round as number,
        name: race.name as string,
        date: race.date as string,
        circuit: race.circuit as string,
        country: (race as any).country || race.circuit as string
      },
      qualifying_results: qualifyingResults as unknown as QualifyingResult[],
      driver_stats: driverStats as unknown as DriverStats[]
    };

    return new Response(JSON.stringify({
      data: response,
      meta: {
        data_source: 'real_f1_data',
        qualifying_results_count: qualifyingResults.length,
        driver_stats_count: driverStats.length,
        last_updated: new Date().toISOString()
      }
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching race features:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      code: 'DATABASE_ERROR'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Get historical patterns for a circuit - REAL DATA ONLY
 */
export async function handleGetCircuitPatternsClean(request: Request, env: Env): Promise<Response> {
  try {
    const url = new URL(request.url);
    const pathParts = url.pathname.split('/');
    const circuit = decodeURIComponent(pathParts[pathParts.length - 2] || '');

    if (!circuit) {
      return new Response(JSON.stringify({ 
        error: 'Circuit name required',
        code: 'MISSING_CIRCUIT_NAME'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const patterns = await env.DB.prepare(`
      SELECT 
        r.circuit,
        r.country,
        COUNT(DISTINCT r.id) as race_count,
        COUNT(DISTINCT r.season) as seasons_held,
        -- Average positions
        AVG(
          CASE 
            WHEN rr.position IS NOT NULL THEN rr.position
            ELSE 21
          END
        ) as avg_finish_position,
        -- DNF rate
        COUNT(CASE WHEN rr.position > 20 THEN 1 END) * 1.0 / 
        NULLIF(COUNT(rr.id), 0) as dnf_rate,
        -- Average qualifying vs race position difference
        AVG(
          CASE 
            WHEN rr.position IS NOT NULL AND qr.grid_position IS NOT NULL 
            THEN rr.position - qr.grid_position
            ELSE NULL
          END
        ) as avg_position_change,
        -- Fastest average qualifying time
        MIN(qr.best_time_ms) as fastest_qualifying_time_ms
      FROM races r
      LEFT JOIN race_results rr ON r.id = rr.race_id
      LEFT JOIN qualifying_results qr ON r.id = qr.race_id AND rr.driver_id = qr.driver_id
      WHERE r.circuit = ?
      GROUP BY r.circuit, r.country
      HAVING COUNT(DISTINCT r.id) > 0
    `).bind(circuit).first();

    if (!patterns) {
      return new Response(JSON.stringify({ 
        error: `No historical data available for circuit '${circuit}'`,
        code: 'CIRCUIT_NO_DATA',
        meta: {
          circuit: circuit,
          suggestion: 'Check circuit name spelling or try a different circuit'
        }
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      data: patterns,
      meta: {
        data_source: 'real_f1_data',
        last_updated: new Date().toISOString()
      }
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching circuit patterns:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      code: 'DATABASE_ERROR'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Batch endpoint to get all data needed for ML prediction - REAL DATA ONLY
 */
export async function handleGetMLPredictionDataClean(request: Request, env: Env): Promise<Response> {
  try {
    const { raceId } = await request.json() as { raceId: number };

    if (!raceId || isNaN(Number(raceId))) {
      return new Response(JSON.stringify({ 
        error: 'Valid race ID required',
        code: 'INVALID_RACE_ID'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Check if race exists
    const race = await env.DB.prepare('SELECT * FROM races WHERE id = ?').bind(raceId).first();
    
    if (!race) {
      return new Response(JSON.stringify({ 
        error: `Race with ID ${raceId} not found`,
        code: 'RACE_NOT_FOUND'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Get all data in parallel
    const [driverStats, qualifyingResults, recentResults] = await Promise.all([
      // Driver statistics
      env.DB.prepare(`
        SELECT 
          d.*,
          te.team_name,
          -- Recent form
          (
            SELECT AVG(
              CASE 
                WHEN rr.position IS NOT NULL THEN rr.position
                ELSE 21
              END
            ) 
            FROM race_results rr
            JOIN races r ON rr.race_id = r.id
            WHERE rr.driver_id = d.id
            AND r.id < ?
            ORDER BY r.date DESC
            LIMIT 3
          ) as avg_recent_position,
          -- Season points
          COALESCE((
            SELECT SUM(points) 
            FROM race_results rr 
            JOIN races r ON rr.race_id = r.id 
            WHERE rr.driver_id = d.id 
            AND r.season = ?
            AND r.id < ?
          ), 0) as season_points
        FROM drivers d
        LEFT JOIN team_entries te ON d.id = te.driver_id AND te.season = ?
      `).bind(raceId, race.season, raceId, race.season).all(),

      // Qualifying results for this race (if available)
      env.DB.prepare(`
        SELECT * FROM qualifying_results WHERE race_id = ?
      `).bind(raceId).all(),

      // Recent race results for pattern analysis
      env.DB.prepare(`
        SELECT rr.*, r.circuit, r.date
        FROM race_results rr
        JOIN races r ON rr.race_id = r.id
        WHERE r.id < ? 
        AND r.season >= ?
        ORDER BY r.date DESC 
        LIMIT 60
      `).bind(raceId, (race.season as number) - 1).all()
    ]);

    // Validate we have sufficient data
    if (driverStats.results.length === 0) {
      return new Response(JSON.stringify({ 
        error: 'No driver data available',
        code: 'INSUFFICIENT_DATA'
      }), {
        status: 422,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify({
      data: {
        race,
        drivers: driverStats.results,
        qualifying_results: qualifyingResults.results,
        recent_results: recentResults.results
      },
      meta: {
        data_source: 'real_f1_data',
        drivers_count: driverStats.results.length,
        qualifying_available: qualifyingResults.results.length > 0,
        recent_results_count: recentResults.results.length,
        data_sufficiency: {
          drivers: driverStats.results.length >= 10 ? 'sufficient' : 'limited',
          qualifying: qualifyingResults.results.length > 0 ? 'available' : 'not_available',
          historical: recentResults.results.length >= 20 ? 'sufficient' : 'limited'
        },
        last_updated: new Date().toISOString()
      }
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Error fetching ML prediction data:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      code: 'DATABASE_ERROR'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}