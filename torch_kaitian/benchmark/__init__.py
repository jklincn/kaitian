import os

import docker

from ..config import CUDA_IMAGE, MLU_IMAGE

root = os.path.dirname(os.path.abspath(__file__))


def run_benchmark_inner(device: str) -> float:
    client = docker.from_env()
    parts = device.split(":")
    device_type = parts[0]
    device_index = parts[1]
    print(f"[KaiTian][Info] {device_type} {device_index}: ", end="", flush=True)
    name = f"kaitian_benchmark_{device.replace(':', '_')}"
    try:
        container = client.containers.get(name)
        container.remove(force=True)
    except docker.errors.NotFound:
        pass
    device_requests = None
    devices = None
    match device_type:
        case "cuda":
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=[device_index], capabilities=[["gpu"]]
                )
            ]
            image = CUDA_IMAGE
            volumes = [f"{root}/cuda.py:/cuda.py"]
            command = ["python", "/cuda.py"]
        case "mlu":
            # Compatible with MLU370
            devices = [
                "/dev/cambricon_ctl",
                f"/dev/cambricon_dev{device_index}",
                f"/dev/cambricon_ipcm{device_index}",
            ]
            image = MLU_IMAGE
            volumes = [f"{root}/mlu.py:/mlu.py"]
            command = ["python", "/mlu.py"]
    container_output = client.containers.run(
        name=name,
        remove=True,
        device_requests=device_requests,
        devices=devices,
        shm_size="16G",
        volumes=volumes,
        working_dir="/",
        image=image,
        command=command,
    )
    output_lines = container_output.decode("utf-8").splitlines()
    print(f"output_lines[-1] seconds", flush=True)
    return float(output_lines[-1])
