import os
import time
import functools
import logging
from dotenv import load_dotenv
import redis

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RedisRateLimiter")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_client = None


def get_redis_client():
    """Returns a singleton Redis client instance."""
    global _redis_client
    if _redis_client is None:
        try:
            client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            _redis_client = client
            logger.info(f"Connected to Redis Docker at {REDIS_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {REDIS_URL} ({e}). Rate limiting will pass-through.")
            _redis_client = False
    return _redis_client if _redis_client is not False else None


class RedisRateLimiter:
    """
    Distributed sliding-window rate limiter powered by Redis.
    Uses Redis Sorted Sets (ZADD, ZREMRANGEBYSCORE, ZCARD) to track timestamps.
    """
    def __init__(self, key: str, max_requests: int = 10, window_seconds: int = 60):
        self.key = f"ratelimit:{key}"
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def acquire(self, wait: bool = True) -> bool:
        r = get_redis_client()
        if r is None:
            # Fallback if Redis is unavailable
            return True

        while True:
            now = time.time()
            window_start = now - self.window_seconds
            pipe = r.pipeline()

            # 1. Remove expired timestamps outside the sliding window
            pipe.zremrangebyscore(self.key, 0, window_start)
            # 2. Count requests currently in the window
            pipe.zcard(self.key)
            # 3. Fetch oldest timestamp in window (for calculating sleep time if full)
            pipe.zrange(self.key, 0, 0, withscores=True)

            _, current_count, oldest = pipe.execute()

            if current_count < self.max_requests:
                # Slot available: record this call
                pipe = r.pipeline()
                pipe.zadd(self.key, {str(now): now})
                pipe.expire(self.key, self.window_seconds + 5)
                pipe.execute()
                return True

            if not wait:
                return False

            # Calculate exact wait time until oldest request rolls out of the window
            if oldest:
                oldest_ts = oldest[0][1]
                sleep_time = max(0.1, (oldest_ts + self.window_seconds) - now + 0.05)
            else:
                sleep_time = 1.0

            logger.info(
                f"[Redis Rate Limit] '{self.key}' hit limit ({current_count}/{self.max_requests} in {self.window_seconds}s). "
                f"Waiting {sleep_time:.2f}s for slot..."
            )
            time.sleep(sleep_time)


def redis_rate_limit(key: str, max_requests: int = 10, window_seconds: int = 60):
    """
    Function decorator to enforce Redis-backed distributed rate limits.
    Example:
        @redis_rate_limit('mistral_api', max_requests=5, window_seconds=60)
        def call_mistral(): ...
    """
    limiter = RedisRateLimiter(key=key, max_requests=max_requests, window_seconds=window_seconds)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire(wait=True)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 5,
    initial_delay: float = 3.0,
    backoff_factor: float = 2.0,
    retryable_status_codes: tuple = (429, 500, 502, 503, 504),
):
    """
    Decorator that catches remote 429/503 HTTP errors and retries
    with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    err_msg = str(exc)

                    is_rate_limit = (
                        status_code in retryable_status_codes
                        or "429" in err_msg
                        or "503" in err_msg
                        or "rate limit" in err_msg.lower()
                        or "temporarily unavailable" in err_msg.lower()
                    )

                    if not is_rate_limit or attempt == max_retries:
                        raise exc

                    logger.warning(
                        f"[Redis Rate Limiter - Retry] Attempt {attempt}/{max_retries} failed for '{func.__name__}'. "
                        f"Retrying in {delay:.1f}s... (Error: {exc})"
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            return func(*args, **kwargs)
        return wrapper
    return decorator


def sleep_between_calls(seconds: float = 0.0):
    """Deprecated: Kept for backwards compatibility."""
    if seconds > 0:
        time.sleep(seconds)
