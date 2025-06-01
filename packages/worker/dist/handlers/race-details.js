import { FastF1Client } from '../services/fastf1-client';
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
        // Initialize FastF1 client
        const fastf1Client = new FastF1Client(env);
        // Get predictions for this race
        const { results: predictions } = await env.DB.prepare(`
      SELECT p.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
      FROM predictions p
      JOIN drivers d ON p.driver_id = d.id
      WHERE p.race_id = ?
      ORDER BY p.predicted_position
    `).bind(raceId).all();
        // Check if we have stored results first
        let results = [];
        const { results: storedResults } = await env.DB.prepare(`
      SELECT rr.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
      FROM race_results rr
      JOIN drivers d ON rr.driver_id = d.id
      WHERE rr.race_id = ?
      ORDER BY rr.position
    `).bind(raceId).all();
        // If no stored results and race is completed, fetch from FastF1
        if (storedResults.length === 0 && new Date(race.date) < new Date()) {
            try {
                const year = new Date(race.date).getFullYear();
                const liveResults = await fastf1Client.getRaceResults(year, race.circuit);
                // Store the results in database
                for (const result of liveResults) {
                    await env.DB.prepare(`
            INSERT OR REPLACE INTO race_results (race_id, driver_id, position, time, points)
            VALUES (?, ?, ?, ?, ?)
          `).bind(raceId, result.driver_id, result.position, result.time, result.points).run();
                }
                // Fetch the stored results again
                const { results: newResults } = await env.DB.prepare(`
          SELECT rr.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
          FROM race_results rr
          JOIN drivers d ON rr.driver_id = d.id
          WHERE rr.race_id = ?
          ORDER BY rr.position
        `).bind(raceId).all();
                results = newResults;
            }
            catch (error) {
                console.error('Failed to fetch live race results:', error);
                results = storedResults;
            }
        }
        else {
            results = storedResults;
        }
        // Check if we have stored qualifying times
        let qualifying = [];
        const { results: storedQualifying } = await env.DB.prepare(`
      SELECT qt.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
      FROM qualifying_times qt
      JOIN drivers d ON qt.driver_id = d.id
      WHERE qt.race_id = ?
      ORDER BY qt.final_position
    `).bind(raceId).all();
        // If no stored qualifying and race weekend has started, fetch from FastF1
        if (storedQualifying.length === 0 && new Date(race.date).getTime() - Date.now() < 3 * 24 * 60 * 60 * 1000) {
            try {
                const year = new Date(race.date).getFullYear();
                const liveQualifying = await fastf1Client.getQualifyingResults(year, race.circuit);
                // Store the qualifying times in database
                for (const qual of liveQualifying) {
                    await env.DB.prepare(`
            INSERT OR REPLACE INTO qualifying_times (race_id, driver_id, q1_time, q2_time, q3_time, final_position)
            VALUES (?, ?, ?, ?, ?, ?)
          `).bind(raceId, qual.driver_id, qual.q1_time, qual.q2_time, qual.q3_time, qual.final_position).run();
                }
                // Fetch the stored qualifying again
                const { results: newQualifying } = await env.DB.prepare(`
          SELECT qt.*, d.code as driver_code, d.name as driver_name, d.team as driver_team
          FROM qualifying_times qt
          JOIN drivers d ON qt.driver_id = d.id
          WHERE qt.race_id = ?
          ORDER BY qt.final_position
        `).bind(raceId).all();
                qualifying = newQualifying;
            }
            catch (error) {
                console.error('Failed to fetch live qualifying results:', error);
                qualifying = storedQualifying;
            }
        }
        else {
            qualifying = storedQualifying;
        }
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
    // Circuit coordinates mapping - complete F1 calendar
    const coordinates = {
        'Australia': { lat: -37.8497, lon: 144.9680 },
        'China': { lat: 31.3389, lon: 121.2199 },
        'Japan': { lat: 34.8434, lon: 136.5408 },
        'Bahrain': { lat: 26.0325, lon: 50.5106 },
        'Saudi Arabia': { lat: 21.6319, lon: 39.1044 },
        'Miami': { lat: 25.9581, lon: -80.2389 },
        'Emilia Romagna': { lat: 44.3439, lon: 11.7167 },
        'Monaco': { lat: 43.7347, lon: 7.4206 },
        'Spain': { lat: 41.5700, lon: 2.2611 },
        'Canada': { lat: 45.5017, lon: -73.5228 },
        'Austria': { lat: 47.2197, lon: 14.7647 },
        'Great Britain': { lat: 52.0786, lon: -1.0169 },
        'Hungary': { lat: 47.5789, lon: 19.2486 },
        'Belgium': { lat: 50.4372, lon: 5.9714 },
        'Netherlands': { lat: 52.3888, lon: 4.5409 },
        'Italy': { lat: 45.6156, lon: 9.2811 },
        'Singapore': { lat: 1.2914, lon: 103.8644 },
        'United States': { lat: 30.1328, lon: -97.6411 },
        'Mexico': { lat: 19.4042, lon: -99.0907 },
        'Brazil': { lat: -23.7036, lon: -46.6997 },
        'Las Vegas': { lat: 36.1147, lon: -115.1728 },
        'Qatar': { lat: 25.4901, lon: 51.4542 },
        'Abu Dhabi': { lat: 24.4672, lon: 54.6031 }
    };
    const coord = coordinates[circuit];
    if (!coord) {
        return { temperature: 20, rain_probability: 0.1, conditions: 'Unknown' };
    }
    try {
        // Check if date is in the past (use historical API) or future (use forecast API)
        const raceDate = new Date(date);
        const today = new Date();
        const daysDiff = Math.floor((raceDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
        let apiUrl;
        if (daysDiff < -5) {
            // Historical weather data
            const dateStr = raceDate.toISOString().split('T')[0];
            apiUrl = `https://archive-api.open-meteo.com/v1/era5?latitude=${coord.lat}&longitude=${coord.lon}&start_date=${dateStr}&end_date=${dateStr}&hourly=temperature_2m,precipitation`;
        }
        else {
            // Forecast data (up to 16 days)
            apiUrl = `https://api.open-meteo.com/v1/forecast?latitude=${coord.lat}&longitude=${coord.lon}&hourly=temperature_2m,precipitation_probability&timezone=auto`;
        }
        const response = await fetch(apiUrl);
        if (response.ok) {
            const data = await response.json();
            const raceHour = 14; // Assume race at 2 PM local time
            // Find the closest hour to race time
            let temperature = 20;
            let rainProb = 0.1;
            if (data.hourly && data.hourly.time) {
                const hourIndex = data.hourly.time.findIndex((t) => {
                    const hourDate = new Date(t);
                    return hourDate.toDateString() === raceDate.toDateString() &&
                        hourDate.getHours() === raceHour;
                });
                if (hourIndex >= 0) {
                    temperature = data.hourly.temperature_2m[hourIndex];
                    rainProb = daysDiff < -5
                        ? (data.hourly.precipitation[hourIndex] > 0 ? 0.8 : 0.1)
                        : data.hourly.precipitation_probability[hourIndex] / 100;
                }
            }
            return {
                temperature: Math.round(temperature),
                rain_probability: rainProb,
                conditions: rainProb > 0.5 ? 'Wet' : rainProb > 0.3 ? 'Variable' : 'Dry'
            };
        }
        // Fallback to typical weather patterns
        return getTypicalWeather(circuit);
    }
    catch (error) {
        console.error('Weather API error:', error);
        return getTypicalWeather(circuit);
    }
}
function getTypicalWeather(circuit) {
    const patterns = {
        'Australia': { temp: 25, rain: 0.15, conditions: 'Warm' },
        'China': { temp: 20, rain: 0.25, conditions: 'Variable' },
        'Japan': { temp: 18, rain: 0.35, conditions: 'Variable' },
        'Bahrain': { temp: 28, rain: 0.05, conditions: 'Hot & Dry' },
        'Saudi Arabia': { temp: 30, rain: 0.05, conditions: 'Hot & Dry' },
        'Miami': { temp: 28, rain: 0.3, conditions: 'Hot & Humid' },
        'Emilia Romagna': { temp: 22, rain: 0.2, conditions: 'Mild' },
        'Monaco': { temp: 24, rain: 0.15, conditions: 'Mediterranean' },
        'Spain': { temp: 26, rain: 0.1, conditions: 'Warm & Dry' },
        'Canada': { temp: 22, rain: 0.25, conditions: 'Variable' },
        'Austria': { temp: 20, rain: 0.3, conditions: 'Mountain Weather' },
        'Great Britain': { temp: 18, rain: 0.4, conditions: 'Variable' },
        'Hungary': { temp: 28, rain: 0.2, conditions: 'Hot' },
        'Belgium': { temp: 19, rain: 0.45, conditions: 'Variable' },
        'Netherlands': { temp: 20, rain: 0.35, conditions: 'Variable' },
        'Italy': { temp: 27, rain: 0.15, conditions: 'Warm' },
        'Singapore': { temp: 30, rain: 0.5, conditions: 'Hot & Humid' },
        'United States': { temp: 32, rain: 0.2, conditions: 'Hot' },
        'Mexico': { temp: 22, rain: 0.3, conditions: 'High Altitude' },
        'Brazil': { temp: 26, rain: 0.4, conditions: 'Variable' },
        'Las Vegas': { temp: 18, rain: 0.05, conditions: 'Desert Night' },
        'Qatar': { temp: 25, rain: 0.05, conditions: 'Desert' },
        'Abu Dhabi': { temp: 27, rain: 0.05, conditions: 'Desert' }
    };
    const pattern = patterns[circuit] || { temp: 20, rain: 0.2, conditions: 'Moderate' };
    return {
        temperature: pattern.temp,
        rain_probability: pattern.rain,
        conditions: pattern.conditions
    };
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
