import time
import threading
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from backend.app.core.config import settings

class RateLimiter:
    def __init__(self, times: int, seconds: int):
        self.times = times
        self.seconds = seconds
        self.history: Dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def __call__(self, request: Request):
        if not settings.RATE_LIMIT_ENABLED:
            return

        # Simple IP-based keying for now
        # In a real app behind proxy, might use request.headers.get("X-Forwarded-For")
        client_ip = request.client.host if request.client else "127.0.0.1"
        key = f"{client_ip}:{request.url.path}"
        
        now = time.time()
        
        with self.lock:
            # Clean up old timestamps
            timestamps = self.history.get(key, [])
            timestamps = [ts for ts in timestamps if now - ts < self.seconds]
            
            if len(timestamps) >= self.times:
                raise HTTPException(
                    status_code=429,
                    detail="Too Many Requests",
                    headers={"Retry-After": str(self.seconds)}
                )
                
            timestamps.append(now)
            self.history[key] = timestamps

def parse_rate_limit(rate_limit_str: str) -> Tuple[int, int]:
    """Parse format like '30/minute' or '5/second'"""
    try:
        parts = rate_limit_str.split("/")
        times = int(parts[0])
        unit = parts[1].lower()
        if unit == "minute" or unit == "minutes":
            seconds = 60
        elif unit == "second" or unit == "seconds":
            seconds = 1
        elif unit == "hour" or unit == "hours":
            seconds = 3600
        else:
            seconds = 60
        return times, seconds
    except Exception:
        return 30, 60

# Configured instances
predict_times, predict_secs = parse_rate_limit(settings.RATE_LIMIT_PREDICT)
feedback_times, feedback_secs = parse_rate_limit(settings.RATE_LIMIT_FEEDBACK)
auth_times, auth_secs = parse_rate_limit(settings.RATE_LIMIT_AUTH)

rate_limit_predict = RateLimiter(times=predict_times, seconds=predict_secs)
rate_limit_feedback = RateLimiter(times=feedback_times, seconds=feedback_secs)
rate_limit_auth = RateLimiter(times=auth_times, seconds=auth_secs)
