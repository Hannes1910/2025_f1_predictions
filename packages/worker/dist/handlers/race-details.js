export async function handleRaceDetails(request, env) {
    try {
        const url = new URL(request.url);
        const raceId = url.pathname.split('/').pop();
        if (!raceId || isNaN(Number(raceId))) {
            return new Response(JSON.stringify({ error: 'Valid race ID required' }), {
                status: 400,
                headers: { 'Content-Type': 'application/json' },
            });
        }
        // Get race information
        const { results: raceResults } = await env.DB.prepare(`
      SELECT r.*, COUNT(p.id) as prediction_count, COUNT(rr.id) as result_count
      FROM races r
      LEFT JOIN predictions p ON r.id = p.race_id
      LEFT JOIN race_results rr ON r.id = rr.race_id
      WHERE r.id = ?
      GROUP BY r.id
    `).bind(raceId).all();
        if (raceResults.length === 0) {
            return new Response(JSON.stringify({ error: 'Race not found' }), {
                status: 404,
                headers: { 'Content-Type': 'application/json' },
            });
        }
        const race = raceResults[0];
        // Get predictions for this race
        const { results: predictions } = await env.DB.prepare(`
      SELECT p.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
      FROM predictions p
      JOIN drivers d ON p.driver_id = d.id
      WHERE p.race_id = ?
      ORDER BY p.predicted_position
    `).bind(raceId).all();
        // Get race results if available
        const { results: results } = await env.DB.prepare(`
      SELECT rr.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
      FROM race_results rr
      JOIN drivers d ON rr.driver_id = d.id
      WHERE rr.race_id = ?
      ORDER BY rr.position
    `).bind(raceId).all();
        // Get qualifying times if available
        const { results: qualifying } = await env.DB.prepare(`
      SELECT qt.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
      FROM qualifying_times qt
      JOIN drivers d ON qt.driver_id = d.id
      WHERE qt.race_id = ?
      ORDER BY qt.final_position
    `).bind(raceId).all();
        // Calculate prediction accuracy if both predictions and results exist
        let accuracy = null;
        if (predictions.length > 0 && results.length > 0) {
            let correctPredictions = 0;
            let totalPositionError = 0;
            for (const prediction of predictions) {
                const actualResult = results.find(r => r.driver_id === prediction.driver_id);
                if (actualResult) {
                    const positionError = Math.abs(Number(prediction.predicted_position) - Number(actualResult.position));
                    totalPositionError += positionError;
                    if (positionError <= 2) { // Within 2 positions = "correct"
                        correctPredictions++;
                    }
                }
            }
            accuracy = {
                correct_predictions: correctPredictions,
                total_predictions: predictions.length,
                accuracy_percentage: (correctPredictions / predictions.length) * 100,
                average_position_error: totalPositionError / predictions.length
            };
        }
        // Get weather data for the race
        const weather = await getWeatherForRace(String(race.circuit), String(race.date));
        // Get feature explanations
        const { results: features } = await env.DB.prepare(`
      SELECT * FROM feature_explanations ORDER BY importance DESC
    `).all();
        return new Response(JSON.stringify({
            race,
            predictions,
            results,
            qualifying,
            accuracy,
            weather,
            features
        }), {
            headers: { 'Content-Type': 'application/json' },
        });
    }
    catch (error) {
        console.error('Error fetching race details:', error);
        return new Response(JSON.stringify({ error: 'Failed to fetch race details' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
        });
    }
}
async function getWeatherForRace(circuit, date) {
    // Circuit coordinates mapping
    const coordinates = {
        'Australia': { lat: -37.8497, lon: 144.9680 },
        'China': { lat: 31.3389, lon: 121.2199 },
        'Japan': { lat: 34.8434, lon: 136.5408 },
        'Spain': { lat: 41.5700, lon: 2.2611 },
        'Canada': { lat: 45.5017, lon: -73.5228 },
        'Austria': { lat: 47.2197, lon: 14.7647 },
        'Monaco': { lat: 43.7347, lon: 7.4206 }
    };
    const coord = coordinates[circuit];
    if (!coord) {
        return { temperature: 20, rain_probability: 0.1, conditions: 'Unknown' };
    }
    try {
        // For past races, return historical weather patterns
        // For future races, we could call the weather API
        const circuitPatterns = {
            'Australia': { temp: 22, rain: 0.2, conditions: 'Mild' },
            'China': { temp: 18, rain: 0.25, conditions: 'Cool' },
            'Japan': { temp: 20, rain: 0.3, conditions: 'Variable' },
            'Spain': { temp: 25, rain: 0.1, conditions: 'Warm & Dry' },
            'Canada': { temp: 22, rain: 0.2, conditions: 'Mild' },
            'Austria': { temp: 19, rain: 0.15, conditions: 'Mountain Weather' },
            'Monaco': { temp: 24, rain: 0.15, conditions: 'Mediterranean' }
        };
        const pattern = circuitPatterns[circuit] ||
            { temp: 20, rain: 0.1, conditions: 'Moderate' };
        return {
            temperature: pattern.temp,
            rain_probability: pattern.rain,
            conditions: pattern.conditions
        };
    }
    catch (error) {
        return { temperature: 20, rain_probability: 0.1, conditions: 'Unknown' };
    }
}
export async function handleQualifyingData(request, env) {
    try {
        const url = new URL(request.url);
        const raceId = url.pathname.split('/')[3]; // /api/qualifying/{raceId}
        const { results: qualifying } = await env.DB.prepare(`
      SELECT qt.*, d.code as driver_code, d.name as driver_name, d.team as driver_team,
             r.name as race_name, r.date as race_date
      FROM qualifying_times qt
      JOIN drivers d ON qt.driver_id = d.id
      JOIN races r ON qt.race_id = r.id
      WHERE qt.race_id = ?
      ORDER BY qt.final_position
    `).bind(raceId).all();
        return new Response(JSON.stringify({ qualifying }), {
            headers: { 'Content-Type': 'application/json' },
        });
    }
    catch (error) {
        console.error('Error fetching qualifying data:', error);
        return new Response(JSON.stringify({ error: 'Failed to fetch qualifying data' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
        });
    }
}
