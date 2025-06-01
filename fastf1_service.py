#!/usr/bin/env python3
"""
FastF1 Data Service
Provides real F1 data through a REST API
"""

import fastf1
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List, Dict
from datetime import datetime
import os

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

app = FastAPI(title="FastF1 Data Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FastF1 Data Service"}

@app.get("/qualifying/{year}/{round}")
async def get_qualifying_results(year: int, round: int):
    """Get qualifying results for a specific race"""
    try:
        # Load the qualifying session
        session = fastf1.get_session(year, round, 'Q')
        session.load()
        
        # Get the results
        results = session.results
        
        qualifying_data = []
        for _, driver in results.iterrows():
            q1_time = driver['Q1'] if pd.notna(driver['Q1']) else None
            q2_time = driver['Q2'] if pd.notna(driver['Q2']) else None
            q3_time = driver['Q3'] if pd.notna(driver['Q3']) else None
            
            # Convert timedelta to string format
            if q1_time:
                q1_time = f"{int(q1_time.total_seconds() // 60)}:{q1_time.total_seconds() % 60:.3f}"
            if q2_time:
                q2_time = f"{int(q2_time.total_seconds() // 60)}:{q2_time.total_seconds() % 60:.3f}"
            if q3_time:
                q3_time = f"{int(q3_time.total_seconds() // 60)}:{q3_time.total_seconds() % 60:.3f}"
            
            qualifying_data.append({
                'driver_code': driver['Abbreviation'],
                'driver_number': driver['DriverNumber'],
                'team': driver['TeamName'],
                'position': driver['Position'],
                'q1_time': q1_time,
                'q2_time': q2_time,
                'q3_time': q3_time
            })
        
        return {
            'year': year,
            'round': round,
            'session': 'Qualifying',
            'circuit': session.event['EventName'],
            'date': session.event['EventDate'].isoformat(),
            'results': qualifying_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/race/{year}/{round}")
async def get_race_results(year: int, round: int):
    """Get race results for a specific race"""
    try:
        # Load the race session
        session = fastf1.get_session(year, round, 'R')
        session.load()
        
        # Get the results
        results = session.results
        
        race_data = []
        winner_time = None
        
        for _, driver in results.iterrows():
            time_str = str(driver['Time']) if pd.notna(driver['Time']) else None
            
            # The winner's time is the full race time
            if driver['Position'] == 1 and time_str:
                winner_time = time_str
            
            race_data.append({
                'driver_code': driver['Abbreviation'],
                'driver_number': driver['DriverNumber'],
                'team': driver['TeamName'],
                'position': driver['Position'],
                'time': time_str,
                'points': driver['Points'],
                'laps': driver['LapsCompleted'],
                'status': driver['Status'],
                'grid_position': driver['GridPosition']
            })
        
        return {
            'year': year,
            'round': round,
            'session': 'Race',
            'circuit': session.event['EventName'],
            'date': session.event['EventDate'].isoformat(),
            'laps': session.event['EventFormat']['Race'],
            'results': race_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calendar/{year}")
async def get_calendar(year: int):
    """Get the F1 calendar for a specific year"""
    try:
        schedule = fastf1.get_event_schedule(year)
        
        events = []
        for _, event in schedule.iterrows():
            events.append({
                'round': event['RoundNumber'],
                'name': event['EventName'],
                'circuit': event['Location'],
                'country': event['Country'],
                'date': event['EventDate'].isoformat() if pd.notna(event['EventDate']) else None,
                'format': event['EventFormat']
            })
        
        return {
            'year': year,
            'total_rounds': len(events),
            'events': events
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live/timing")
async def get_live_timing():
    """Get live timing data (if available)"""
    try:
        # This would connect to live timing feed
        # For now, return a placeholder
        return {
            'status': 'no_session',
            'message': 'No live session currently active'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import pandas as pd

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("fastf1_service:app", host="0.0.0.0", port=port, reload=True)