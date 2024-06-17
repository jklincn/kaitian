import json
import os
import shutil
import subprocess
from datetime import datetime

import docker
import tomlkit

USE_CUDA = True if os.environ.get("USE_CUDA", "1") == "1" else False
USE_MLU = True if os.environ.get("USE_MLU", "1") == "1" else False

CUDA_BASE_IMAGE = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel"
CUDA_IMAGE = "jklincn/kaitian:0.0.0-cuda"
MLU_BASE_IMAGE = (
    "yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310"
)
MLU_IMAGE = "jklincn/kaitian:0.0.0-mlu"


def find_cuda(config_data):
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
        cuda_device["name"] = gpu_info["model"]
        cuda_device["bus_id"] = gpu_info["bus_id"]
        cuda_device["link_status"] = (
            f"PCIe Gen{gpu_info['current_pcie_link_gen']} x{gpu_info['current_pcie_link_width']} ({gpu_info['link_ok']})"
        )
        cuda_device["memory"] = gpu_info["memory_total"]
        cuda.add(f"cuda{i}", cuda_device)

    config_data["devices"].add("cuda", cuda)
    # device_counts["CUDA"] = len(gpu_info_list)
    return len(gpu_info_list)


def find_mlu(config_data):
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
    # device_counts["MLU"] = count
    config_data["devices"].add("mlu", mlu)
    return count


def check_cuda():
    client = docker.from_env()
    try:
        client.images.get(CUDA_BASE_IMAGE)
    except docker.errors.ImageNotFound:
        print(f"Image {CUDA_BASE_IMAGE} does not exist.")
        print(f"You must `docker pull {CUDA_BASE_IMAGE}` first.")
        exit()
    try:
        client.images.get(CUDA_IMAGE)
    except docker.errors.ImageNotFound:
        print(f"Image {CUDA_IMAGE} does not exist.")
        print(f"Building {CUDA_IMAGE}...")
        build_command = [
            "docker",
            "build",
            "-t",
            CUDA_IMAGE,
            "--build-arg",
            f"IMAGE={CUDA_BASE_IMAGE}",
            "--build-arg",
            "DEVICE=CUDA",
            os.path.dirname(os.path.abspath(__file__)),
        ]
        process = subprocess.Popen(
            build_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                exit_code = process.poll()
                if exit_code != 0:
                    exit("image build error: ", exit_code)
                break
            if output:
                print(output.strip(), flush=True)

    print(f"Use CUDA Image: {CUDA_IMAGE}")
    # device_types.append("CUDA")


def check_mlu():
    client = docker.from_env()
    try:
        client.images.get(MLU_BASE_IMAGE)
    except docker.errors.ImageNotFound:
        print(f"Image {MLU_BASE_IMAGE} does not exist.")
        print(
            f"You must get {MLU_BASE_IMAGE} first. See https://sdk.cambricon.com/download?component_name=PyTorch"
        )
        exit()
    try:
        client.images.get(MLU_IMAGE)
    except docker.errors.ImageNotFound:
        print(f"Image {MLU_IMAGE} does not exist.")
        print(f"Building {MLU_IMAGE}...")
        build_command = [
            "docker",
            "build",
            "-t",
            MLU_IMAGE,
            "--build-arg",
            f"IMAGE={MLU_BASE_IMAGE}",
            "--build-arg",
            "DEVICE=MLU",
            os.path.dirname(os.path.abspath(__file__)),
        ]
        process = subprocess.Popen(
            build_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                exit_code = process.poll()
                if exit_code != 0:
                    exit("image build error: ", exit_code)
                break
            if output:
                print(output.strip(), flush=True)

    print(f"Use MLU Image: {MLU_IMAGE}")
    # device_types.append("MLU")


def find_device(config_data):
    print("-----------------------------")
    if USE_CUDA:
        cuda_count = find_cuda(config_data)
        if cuda_count > 0:
            # check_cuda()
            print("-----------------------------")
    if USE_MLU:
        mlu_count = find_mlu(config_data)
        if mlu_count > 0:
            # check_mlu()
            print("-----------------------------")
    # for device_type in device_types:
    #     total_nums += device_counts[device_type]


def init_kaitian(args):

    # 定义配置文件路径
    config_dir = os.path.join(os.path.expanduser("~"), ".config", "kaitian")
    config_file = os.path.join(config_dir, "kaitian.toml")

    # 确保配置目录存在
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # 定义初始配置内容
    config_data = tomlkit.document()
    creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_data.add("create_time", creation_time)
    config_data.add("devices", tomlkit.table())

    # 如果配置文件不存在，则创建并写入初始数据
    if args.force or not os.path.isfile(config_file):
        find_device(config_data)
        with open(config_file, "w") as file:
            file.write(tomlkit.dumps(config_data))
    else:
        with open(config_file, "r") as file:
            data = tomlkit.loads(file.read())
            created_time = data.get("create_time")
            created_datetime = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")
            formatted_time = created_datetime.strftime("%H:%M on June %d, %Y")
            print(
                f"[KaiTian][Warning] KaiTian has been initialized at {formatted_time}. You can use '-f' or '--force' to reinitialize."
            )
