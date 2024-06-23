import multiprocessing
import os
import subprocess
import traceback
from pathlib import Path

import docker
import tomlkit

from .. import config
from . import monitor, redis


def run_container(
    device_type: str,
    device_ids: list[str],
    args,
    gloo_rank: int,
    gloo_world_size: int,
    global_world_size: int,
    global_rank_start: int,
    unknown_args,
):
    print(f"[KaiTian][Info] Running {device_type} container...")
    client = docker.from_env()
    environment = {
        "KAITIAN_GLOO_RANK": gloo_rank,
        "KAITIAN_GLOO_WORLD_SIZE": gloo_world_size,
        "KAITIAN_GLOBAL_WORLD_SIZE": global_world_size,
        "KAITIAN_GLOBAL_RANK_START": global_rank_start,
    }
    if "wait" in args.develop:
        kaitian_path = Path(__file__).resolve().parent.parent.parent
        volumes = [f"{kaitian_path}:/kaitian"]
        command = None
    else:
        file_path = Path(args.file).resolve()
        volumes = [
            f"{file_path}:/{file_path.name}",
            f"{file_path.parent}/data:/data",
            f"/home/lin/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints",
        ]
        command = ["python", f"/{file_path.name}"] + unknown_args
    device_requests = None
    devices = None
    match device_type:
        case "cuda":
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=device_ids, capabilities=[["gpu"]]
                )
            ]
            image = config.CUDA_IMAGE
        case "mlu":
            # Compatible with MLU370
            devices = ["/dev/cambricon_ctl"]
            for i in device_ids:
                devices.extend([f"/dev/cambricon_dev{i}", f"/dev/cambricon_ipcm{i}"])
            image = config.MLU_IMAGE
    return client.containers.run(
        detach=True,
        network="kaitian",
        name=f"kaitian_{device_type}",
        hostname=device_type,
        environment=environment,
        device_requests=device_requests,
        devices=devices,
        shm_size="16G",
        volumes=volumes,
        working_dir="/",
        image=image,
        command=command,
    )


def stream_logs(container, device_type):
    try:
        logs = []
        for line in container.logs(stream=True, follow=True):
            logs.append(line.decode().strip())
            print(f"[{device_type}] {line.decode().strip()}", flush=True)
    except:
        pass
    finally:
        with open(f"log_{device_type}.txt", "w") as log_file:
            log_file.write("\n".join(logs))


def docker_run(args, unknown_args, device_list: list[str]):
    if len(device_list) == 0:
        exit(f"[KaiTian][Error] No device specified for use.")
    client = docker.from_env()
    containers = {}

    # create network
    try:
        network = client.networks.get("kaitian")
    except docker.errors.NotFound:
        network = client.networks.create("kaitian", driver="bridge")

    # create monitor
    if "wait" not in args.develop:
        monitor_stop_flag = multiprocessing.Value("b", False)
        monitor_process = multiprocessing.Process(
            target=monitor.run_monitor, args=(monitor_stop_flag,)
        )
        monitor_process.start()

    # create redis server
    client.containers.run(
        detach=True,
        network="kaitian",
        name=f"kaitian_redis",
        image=config.REDIS_IMAGE,
        command="redis-server",
    )

    # create accelerator container
    device_types = set(device.split(":")[0] for device in device_list)
    gloo_world_size = len(device_types)
    global_world_size = len(device_list)
    global_rank_start = 0
    try:
        for gloo_rank, device_type in enumerate(device_types):
            devices = [device for device in device_list if device_type in device]
            redis.set_capability(global_rank_start, devices, device_type)
            device_ids = [device.split(":")[1] for device in devices]
            containers[device_type] = run_container(
                device_type,
                device_ids,
                args,
                gloo_rank,
                gloo_world_size,
                global_world_size,
                global_rank_start,
                unknown_args,
            )
            device_count = sum(
                1 for device in device_list if device.startswith(device_type)
            )
            global_rank_start += device_count
        if "wait" in args.develop:
            print("[KaiTian][Info] Detected 'wait' development argument.", flush=True)
            print(
                "[KaiTian][Info] You can use 'docker exec' to enter the container.",
                flush=True,
            )
            print(
                "[KaiTian][Info] And You should manually remove the containers and network.",
                flush=True,
            )
            return
        # get output
        print("--------- Output -----------")
        log_processes = []
        for device_type, container in containers.items():
            process = multiprocessing.Process(
                target=stream_logs, args=(container, device_type)
            )
            process.start()
            log_processes.append(process)
        for process in log_processes:
            process.join()
        print("--------- Finish -----------")
        for device_type in device_types:
            print(
                f"{device_type.upper()} log: {Path.cwd()}/log_{device_type}.txt",
                flush=True,
            )
    except KeyboardInterrupt:
        print(
            "\n[KaiTian][Info] Received KeyboardInterrupt. Stop and clean up.",
            flush=True,
        )
    except Exception as e:
        print(f"[KaiTian][Error] Unknown error: {e}", flush=True)
        tb = traceback.format_exc()
        print(tb)
    finally:
        if "wait" in args.develop:
            return
        # clean up
        all_containers = client.containers.list(all=True)
        container_names = [f"kaitian_{device_type}" for device_type in device_types]
        for container in all_containers:
            container_name = container.attrs["Name"].strip("/")
            if container_name in container_names or container_name == "kaitian_redis":
                container.remove(force=True)
        network.remove()

    if "wait" not in args.develop:
        monitor_stop_flag.value = True
        monitor_process.join()


def check_environment_variable(config_data: tomlkit.TOMLDocument) -> list[str]:

    def check_use_xxx(device_type: str):
        variable = os.environ.get(f"USE_{device_type}", "all")
        try:
            if variable == "-1":
                return
            elif variable == "all":
                devices = config_data["devices"][device_type.lower()]
                device_numbers = [
                    device["device_number"]
                    for device_key, device in devices.items()
                    if device_key.startswith(device_type.lower())
                ]
            else:
                device_numbers = [
                    f"{device_type.lower()}:{id}"
                    for id in map(int, variable.split(","))
                ]
            device_list.extend(device_numbers)
        except ValueError:
            exit(
                f"[KaiTian][Error] The provided USE_{device_type} is invalid: {variable}"
            )

    device_list = []
    check_use_xxx("CUDA")
    check_use_xxx("MLU")
    return device_list


def build_image(device_list: list[str]):

    def build_image_inner(device_type: str):
        match device_type:
            case "cuda":
                base_image = config.CUDA_BASE_IMAGE
                image = config.CUDA_IMAGE
            case "mlu":
                base_image = config.MLU_BASE_IMAGE
                image = config.MLU_IMAGE
            case _:
                exit(
                    f"[KaiTian][Error] Internal error: unmatched device_type {device_type}."
                )
        client = docker.from_env()
        try:
            client.images.get(image)
        except docker.errors.ImageNotFound:
            try:
                client.images.get(base_image)
            except docker.errors.ImageNotFound:
                exit(f"[KaiTian][Error] Base Image {base_image} does not exist.")
            print(f"[KaiTian][Info] Building {image}")
            build_command = [
                "docker",
                "build",
                "-t",
                image,
                "--build-arg",
                f"IMAGE={base_image}",
                "--build-arg",
                f"DEVICE={device_type.upper()}",
                ".",
            ]
            process = subprocess.Popen(
                build_command,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    exit_code = process.poll()
                    if exit_code != 0:
                        exit(f"[KaiTian][Error] Build image error: {exit_code}")
                    break
                if output:
                    print(output.strip(), flush=True)

    device_types = set(device.split(":")[0] for device in device_list)
    for device_type in device_types:
        build_image_inner(device_type)


def run_kaitian(args, unknown_args):
    # get config data
    if not os.path.isfile(config.CONFIG_FILE):
        exit(
            f"[KaiTian][Error] Unable to find configuration file. Please run 'kaitian init' first."
        )
    with open(config.CONFIG_FILE, "r") as file:
        config_data = tomlkit.loads(file.read())

    # get the specified device to use
    device_list = check_environment_variable(config_data)

    if "build" in args.develop:
        build_image(device_list)

    docker_run(args, unknown_args, device_list)
