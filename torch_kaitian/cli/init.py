import json
import os
import shutil
import subprocess
from datetime import datetime

import docker
import tomlkit

from ..benchmark import run_benchmark_inner
from ..config import (
    CONFIG_DIR,
    CONFIG_FILE,
    CUDA_IMAGE,
    MAX_COMPUTE_CAPABILITY,
    MLU_IMAGE,
    REDIS_IMAGE,
)


def find_cuda(config_data: tomlkit.TOMLDocument):
    if shutil.which("nvidia-smi") is None:
        print("[KaiTian][Info] nvidia-smi not found, skip CUDA.")
        return
    result = subprocess.run(
        "nvidia-smi --query-gpu=name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max,memory.total --format=csv,noheader",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = result.stdout.strip().split("\n")

    def parse_gpu_info(gpu_str):
        parts = gpu_str.split(", ")
        gpu_info = {
            "model": parts[0],
            "bus_id": parts[1],
            "current_pcie_link_gen": parts[2],
            "current_pcie_link_width": parts[3],
            "max_pcie_link_gen": parts[4],
            "max_pcie_link_width": parts[5],
            "link_ok": (
                "ok" if parts[2] == parts[4] and parts[3] == parts[5] else "downgraded"
            ),
            "memory_total": parts[6],
        }
        return gpu_info

    gpu_info_list = [parse_gpu_info(line) for line in lines if line.strip()]
    cuda = tomlkit.table()
    cuda.add("image", CUDA_IMAGE)
    for i, gpu_info in enumerate(gpu_info_list):
        cuda_device = tomlkit.table()
        cuda_device["device_number"] = f"cuda:{i}"
        cuda_device["name"] = gpu_info["model"]
        cuda_device["bus_id"] = gpu_info["bus_id"]
        cuda_device["link_status"] = (
            f"PCIe Gen{gpu_info['current_pcie_link_gen']} x{gpu_info['current_pcie_link_width']} ({gpu_info['link_ok']})"
        )
        cuda_device["memory"] = gpu_info["memory_total"]
        cuda.add(f"cuda{i}", cuda_device)
    config_data["devices"].add("cuda", cuda)


def find_mlu(config_data: tomlkit.TOMLDocument):
    if shutil.which("cnmon") is None:
        print("[KaiTian][Info] cnmon not found, skip MLU.")
        return
    subprocess.run(
        "cnmon info -j",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    with open("cnmon_info.json", "r") as f:
        info = json.load(f)
        count = len(info["CnmonInfo"])
    os.remove("cnmon_info.json")
    speed = {
        "2.5 GT/s": "1",
        "5 GT/s": "2",
        "8 GT/s": "3",
        "16 GT/s": "4",
        "32 GT/s": "5",
    }
    mlu = tomlkit.table()
    mlu.add("image", MLU_IMAGE)
    for i in range(count):
        pci = info["CnmonInfo"][i]["PCI"]
        link_ok = (
            "ok"
            if pci["CurrentSpeed"] == pci["MaxSpeed"]
            and pci["CurrentWidth"] == pci["MaxWidth"]
            else "downgraded"
        )
        mlu_device = tomlkit.table()
        mlu_device["device_number"] = f"mlu:{i}"
        mlu_device["name"] = info["CnmonInfo"][i]["ProductName"]
        mlu_device["bus_id"] = (
            f"{pci['DomainID']}:{pci['Bus']}:{pci['Device']}.{pci['Function']}"
        )
        mlu_device["link_status"] = (
            f"PCIe Gen{speed[pci['CurrentSpeed']]} {pci['CurrentWidth']} ({link_ok})"
        )
        mlu_device["memory"] = (
            f"{info['CnmonInfo'][i]['PhysicalMemUsage']['Total']} MiB"
        )
        mlu.add(f"mlu{i}", mlu_device)
    config_data["devices"].add("mlu", mlu)


# find all available devices by default
def find_devices(config_data):
    print("[KaiTian][Info] Looking for available devices")
    find_cuda(config_data)
    find_mlu(config_data)
    # print devices information
    print("-----------------------------")
    for device_type in config_data["devices"]:
        for _, detail in config_data["devices"][device_type].items():
            if isinstance(detail, dict):
                print(f"Device Number: {detail['device_number']}")
                print(f"Name: {detail['name']}")
                print(f"Bus Id: {detail['bus_id']}")
                print(f"Link Status: {detail['link_status']}")
                print(f"Memory: {detail['memory']}")
                print("-----------------------------")


def pull_images(config_data: tomlkit.TOMLDocument):
    print("[KaiTian][Info] Pulling relevant images")
    client = docker.from_env()

    def pull_image_inner(image: str):
        try:
            client.images.get(image)
            print(f"[KaiTian][Info] Image {image} already exists.")
        except docker.errors.ImageNotFound:
            print(f"[KaiTian][Info] Pulling {image}")
            resp = client.api.pull(image, stream=True, decode=True)
            completed_layers = set()
            for line in resp:
                layer_id = line.get("id", "")
                status = line.get("status", "")
                if status == "Pull complete":
                    if layer_id not in completed_layers:
                        print(f"{layer_id}: {status}")
                        completed_layers.add(layer_id)
            print(f"[KaiTian][Info] Successfully pulled {image}.")

    # pull redis
    pull_image_inner(REDIS_IMAGE)

    # pull kaitian image
    for device_type in config_data["devices"]:
        image = config_data["devices"][device_type]["image"]
        pull_image_inner(image)


def run_benchmark(config_data: tomlkit.TOMLDocument):
    print("[KaiTian][Info] Running the benchmark for each device")
    benchmark_result = {}
    for device_type in config_data["devices"]:
        for device, detail in config_data["devices"][device_type].items():
            if isinstance(detail, dict):
                benchmark_result[device] = run_benchmark_inner(detail["device_number"])
    min_time = min(benchmark_result.values())
    for device_type in config_data["devices"]:
        for device, detail in config_data["devices"][device_type].items():
            if isinstance(detail, dict):
                detail["compute_capability"] = round(
                    min_time / benchmark_result[device] * MAX_COMPUTE_CAPABILITY, 1
                )


def create_config() -> tomlkit.TOMLDocument:
    print(f"[KaiTian][Info] Creating configuration file ({CONFIG_FILE})")

    # 1. create empty config and write creation time
    config_data = tomlkit.document()
    creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_data.add("create_time", creation_time)
    config_data.add("devices", tomlkit.table())

    # 2. add device information
    find_devices(config_data)

    # 3. pull images
    pull_images(config_data)

    # 4. run benchmark for each device
    run_benchmark(config_data)

    # 5. done
    return config_data


def init_kaitian(args, unknown_args):
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    if not os.path.isfile(CONFIG_FILE) or args.force:
        # configuration file does not exist or is forcibly overwritten
        config_data = create_config()
        with open(CONFIG_FILE, "w") as file:
            file.write(tomlkit.dumps(config_data))
    else:
        # configuration file already exists
        with open(CONFIG_FILE, "r") as file:
            data = tomlkit.loads(file.read())
            created_time = data.get("create_time")
            created_datetime = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")
            print(
                f"[KaiTian][Warning] KaiTian has already been initialized as of {created_datetime}. Use the '-f' or '--force' option to reinitialize if necessary."
            )
