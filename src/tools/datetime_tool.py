"""
Date and Time Tool.
Provides temporal grounding for current date, time, day of week, and ISO format.
"""
from datetime import datetime, timezone
from typing import Dict, Any

def get_current_datetime(tz: str = "UTC") -> Dict[str, Any]:
    """
    Returns current date, time, and timezone information.
    
    Args:
        tz: Timezone representation (defaults to UTC).
        
    Returns:
        dict containing formatted date, time, day_of_week, and timestamp.
    """
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now()

    return {
        "status": "success",
        "utc_now": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "utc_iso": now_utc.isoformat(),
        "local_now": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "current_year": now_utc.year,
        "current_date": now_utc.strftime("%Y-%m-%d"),
        "day_of_week": now_utc.strftime("%A"),
        "summary": f"Current date is {now_utc.strftime('%A, %B %d, %Y')}, time: {now_utc.strftime('%H:%M:%S UTC')}."
    }
