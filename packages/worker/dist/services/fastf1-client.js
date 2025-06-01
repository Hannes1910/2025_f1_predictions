// Map driver codes to our database IDs
const DRIVER_CODE_TO_ID = {
    'VER': 1, 'HAD': 2, 'NOR': 3, 'PIA': 4, 'LEC': 5,
    'HAM': 6, 'RUS': 7, 'ANT': 8, 'ALO': 9, 'STR': 10,
    'GAS': 11, 'DOO': 12, 'SAI': 13, 'ALB': 14, 'TSU': 15,
    'LAW': 16, 'HUL': 17, 'BEA': 18, 'BOT': 19, 'BOR': 20
};
// Circuit mapping for FastF1
const CIRCUIT_TO_ROUND = {
    'Australia': 1,
    'China': 2,
    'Japan': 3,
    'Bahrain': 4,
    'Saudi Arabia': 5,
    'Miami': 6,
    'Emilia Romagna': 7,
    'Monaco': 8,
    'Spain': 9,
    'Canada': 10,
    'Austria': 11,
    'Great Britain': 12,
    'Hungary': 13,
    'Belgium': 14,
    'Netherlands': 15,
    'Italy': 16,
    'Singapore': 17,
    'United States': 18,
    'Mexico': 19,
    'Brazil': 20,
    'Las Vegas': 21,
    'Qatar': 22,
    'Abu Dhabi': 23
};
export class FastF1Client {
    pythonServiceUrl;
    constructor(env) {
        // You'll need to set up a Python service that wraps FastF1
        // Since FastF1 is a Python library, we need a bridge service
        this.pythonServiceUrl = env.FASTF1_SERVICE_URL || 'http://localhost:8000';
    }
    async getQualifyingResults(year, circuit) {
        const round = CIRCUIT_TO_ROUND[circuit];
        if (!round) {
            throw new Error(`Unknown circuit: ${circuit}`);
        }
        try {
            const response = await fetch(`${this.pythonServiceUrl}/qualifying/${year}/${round}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch qualifying data: ${response.statusText}`);
            }
            const data = await response.json();
            // Transform FastF1 data to our format
            return data.results.map((result) => ({
                driver_id: DRIVER_CODE_TO_ID[result.driver_code],
                driver_code: result.driver_code,
                q1_time: result.q1_time ? this.lapTimeToSeconds(result.q1_time) : null,
                q2_time: result.q2_time ? this.lapTimeToSeconds(result.q2_time) : null,
                q3_time: result.q3_time ? this.lapTimeToSeconds(result.q3_time) : null,
                final_position: result.position
            })).filter((r) => r.driver_id); // Filter out unknown drivers
        }
        catch (error) {
            console.error('Error fetching qualifying data:', error);
            return [];
        }
    }
    async getRaceResults(year, circuit) {
        const round = CIRCUIT_TO_ROUND[circuit];
        if (!round) {
            throw new Error(`Unknown circuit: ${circuit}`);
        }
        try {
            const response = await fetch(`${this.pythonServiceUrl}/race/${year}/${round}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch race data: ${response.statusText}`);
            }
            const data = await response.json();
            // Transform FastF1 data to our format
            return data.results.map((result) => ({
                driver_id: DRIVER_CODE_TO_ID[result.driver_code],
                driver_code: result.driver_code,
                position: result.position,
                time: this.raceTimeToSeconds(result.time),
                points: result.points,
                laps_completed: result.laps,
                status: result.status
            })).filter((r) => r.driver_id); // Filter out unknown drivers
        }
        catch (error) {
            console.error('Error fetching race data:', error);
            return [];
        }
    }
    lapTimeToSeconds(lapTime) {
        // Convert "1:23.456" to seconds
        const parts = lapTime.split(':');
        const minutes = parseInt(parts[0]);
        const seconds = parseFloat(parts[1]);
        return minutes * 60 + seconds;
    }
    raceTimeToSeconds(raceTime) {
        // Convert race time or "+12.345s" format to seconds
        if (raceTime.startsWith('+')) {
            // This is a gap, we'll need the winner's time to calculate absolute time
            return 0; // Placeholder - should be calculated based on winner's time
        }
        // Parse full race time format
        return this.lapTimeToSeconds(raceTime);
    }
}
