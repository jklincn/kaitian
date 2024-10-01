import docker
import redis
import tomlkit

from .. import config

redis_client = None


# CLI environment is not in the kaitian network, So we need to manually find the IP address
def get_redis_client():
    global redis_client
    if redis_client is None:
        client = docker.from_env()
        redis_container = client.containers.get("kaitian_redis")
        ip = redis_container.attrs["NetworkSettings"]["Networks"]["kaitian"][
            "IPAddress"
        ]
        redis_client = redis.Redis(
            host=ip,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=True,
        )
    return redis_client


def set_capability(global_rank_start: int, devices: list[str], device_type: str):
    r = get_redis_client()
    devices = [device.replace(":", "") for device in devices]
    with open(config.CONFIG_FILE, "r") as file:
        config_data = tomlkit.loads(file.read())

    for index, device in enumerate(devices):
        compute_capability = str(
            config_data["devices"][device_type][device]["compute_capability"]
        )
        global_rank = str(global_rank_start + index)
        r.hset("compute_capability", global_rank, compute_capability)
