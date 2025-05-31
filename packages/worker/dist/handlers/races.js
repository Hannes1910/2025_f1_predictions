export async function handleRaces(request, env) {
    try {
        const url = new URL(request.url);
        const season = url.searchParams.get('season') || new Date().getFullYear().toString();
        const { results } = await env.DB.prepare(`SELECT 
        r.*,
        COUNT(DISTINCT p.id) as prediction_count,
        COUNT(DISTINCT rr.id) as result_count,
        CASE 
          WHEN r.date < date('now') THEN 'completed'
          WHEN r.date = date('now') THEN 'today'
          ELSE 'upcoming'
        END as status
       FROM races r
       LEFT JOIN predictions p ON r.id = p.race_id
       LEFT JOIN race_results rr ON r.id = rr.race_id
       WHERE r.season = ?
       GROUP BY r.id
       ORDER BY r.round`).bind(season).all();
        return new Response(JSON.stringify({ races: results }), {
            headers: { 'Content-Type': 'application/json' },
        });
    }
    catch (error) {
        return new Response(JSON.stringify({ error: 'Failed to fetch races' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
        });
    }
}
