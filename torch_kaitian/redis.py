import redis

__all__ = ["get_compute_capability"]

redis_client = None
data = None


def _get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host="kaitian_redis",
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=True,
        )
    return redis_client


def _get_data():
    global data
    if data is None:
        r = _get_redis_client()
        data = r.hgetall("compute_capability")
    return data


def get_compute_capability(global_rank: int) -> float:
    data = _get_data()
    return float(data[str(global_rank)])
