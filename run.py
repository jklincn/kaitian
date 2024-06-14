import argparse
import json
import os
import shutil
import subprocess
import docker
import threading

cuda_nums = 0
mlu_nums = 0
total_nums = 0
device_types = []
coordinator = ""

USE_CUDA = True if os.environ.get("USE_CUDA", "1") == "1" else False
USE_MLU = True if os.environ.get("USE_MLU", "1") == "1" else False

CUDA_BASE_IMAGE = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel"
CUDA_IMAGE = "jklincn/kaitian:0.0.0-cuda"
MLU_BASE_IMAGE = (
    "yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310"
)
MLU_IMAGE = "jklincn/kaitian:0.0.0-mlu"


def find_cuda():
    if shutil.which("nvidia-smi") is None:
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
    for gpu_info in gpu_info_list:
        print(f"Find {gpu_info['model']}")
        print(f"Bus Id: {gpu_info['bus_id']}")
        print(
            f"Link Status: PCIe Gen{gpu_info['current_pcie_link_gen']} x{gpu_info['current_pcie_link_width']} ({gpu_info['link_ok']})"
        )
        print(f"Memory: {gpu_info['memory_total']}")
        print("-----------------------------")
    return len(gpu_info_list)


def find_mlu():
    if shutil.which("cnmon") is None:
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
    for i in range(count):
        pci = info["CnmonInfo"][i]["PCI"]
        speed = {
            "2.5 GT/s": "1",
            "5 GT/s": "2",
            "8 GT/s": "3",
            "16 GT/s": "4",
            "32 GT/s": "5",
        }
        link_ok = (
            "ok"
            if pci["CurrentSpeed"] == pci["MaxSpeed"]
            and pci["CurrentWidth"] == pci["MaxWidth"]
            else "downgraded"
        )
        print(f"Find {info['CnmonInfo'][i]['ProductName']}")
        print(
            f"Bus Id: {pci['DomainID']}:{pci['Bus']}:{pci['Device']}.{pci['Function']}"
        )
        print(
            f"Link Status: PCIe Gen{speed[pci['CurrentSpeed']]} {pci['CurrentWidth']} ({link_ok})"
        )
        print(f"Memory: {info['CnmonInfo'][i]['PhysicalMemUsage']['Total']} MiB")
        print("-----------------------------")
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
    device_types.append("CUDA")


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
    device_types.append("MLU")


def find_device():
    global cuda_nums
    global mlu_nums
    global total_nums
    print("-----------------------------")
    if USE_CUDA:
        cuda_nums = find_cuda()
        if cuda_nums > 0:
            check_cuda()
            print("-----------------------------")
    if USE_MLU:
        mlu_nums = find_mlu()
        if mlu_nums > 0:
            check_mlu()
            print("-----------------------------")
    total_nums = cuda_nums + mlu_nums


def run_container(device_type, file, rank, world_size):
    print(f"Running {device_type} container...")
    client = docker.from_env()
    command = f"python /{os.path.basename(file)}"
    match device_type:
        case "CUDA":
            return client.containers.run(
                detach=True,
                network="kaitian",
                name=device_type,
                environment={
                    "TOTAL_NUMS": total_nums,
                    "KAITIAN_RANK": rank,
                    "KAITIAN_WORLD_SIZE": world_size,
                },
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=["all"], capabilities=[["gpu"]]
                    )
                ],
                shm_size="16G",
                volumes=[
                    "kaitian:/tmp/gloo",
                    f"{os.path.abspath(file)}:/{os.path.basename(file)}",
                    f"{os.path.dirname(os.path.abspath(file))}/data:/data",
                ],
                working_dir="/",
                image=CUDA_IMAGE,
                command=command,
            )
        case "MLU":
            # Compatible with MLU370
            device = ["/dev/cambricon_ctl"]
            for i in range(mlu_nums):
                device.extend([f"/dev/cambricon_dev{i}", f"/dev/cambricon_ipcm{i}"])
            return client.containers.run(
                detach=True,
                network="kaitian",
                name=device_type,
                environment={
                    "TOTAL_NUMS": total_nums,
                    "KAITIAN_RANK": rank,
                    "KAITIAN_WORLD_SIZE": world_size,
                },
                devices=device,
                shm_size="16G",
                volumes=[
                    "kaitian:/tmp/gloo",
                    f"{os.path.abspath(file)}:/{os.path.basename(file)}",
                    f"{os.path.dirname(os.path.abspath(file))}/data:/data",
                ],
                working_dir="/",
                image=MLU_IMAGE,
                command=command,
            )

def stream_logs(container, device_type):
    logs = []
    for line in container.logs(stream=True, follow=True):
        logs.append(line.decode().strip())
        print(f"[{device_type}] {line.decode().strip()}", flush=True)
    with open(f"log_{device_type}.txt", "w") as log_file:
        log_file.write("\n".join(logs))
    

def docker_run(file):
    global coordinator
    client = docker.from_env()
    containers = {}
    if len(device_types) == 0:
        exit("No device available.")
    # create network
    try:
        network = client.networks.get("kaitian")
    except docker.errors.NotFound:
        network = client.networks.create("kaitian", driver="bridge")
    # create volume
    try:
        volume = client.volumes.get("kaitian")
        volume.remove(force=True)
    except docker.errors.NotFound:
        pass
    finally:
        volume = client.volumes.create(
            name="kaitian",
            driver="local",
            driver_opts={"type": "tmpfs", "device": "tmpfs", "o": "size=100m"},
        )
    try:
        coordinator = device_types[0]
        # create container
        for rank, device_type in enumerate(device_types):
            containers[device_type] = run_container(
                device_type, file, rank, len(device_types)
            )
        print("--------- Output -----------")
        threads = []
        for device_type, container in containers.items():
            thread = threading.Thread(target=stream_logs, args=(container, device_type))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        print("--------- Finish -----------")
        for device_type in device_types:
            print(f"{device_type} log file path: {os.path.dirname(os.path.abspath(__file__))}/log_{device_type}.txt", flush=True)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Stop and remove all containers.")
    finally:
        for device_type in device_types:
            try:
                node = client.containers.get(device_type)
                node.remove(force=True)
            except docker.errors.NotFound:
                continue
        network.remove()
        volume.remove()


def get_args():
    parser = argparse.ArgumentParser(description="KaiTian Launcher")
    parser.add_argument(
        "FILE", help="Your training code, for example: python run.py train.py"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="enable quiet mode, less output is printed",
    )
    args = parser.parse_args()

    if os.path.exists(args.FILE):
        return args.FILE
    else:
        raise FileNotFoundError(f"{args.FILE} not found.")


if __name__ == "__main__":
    file = get_args()
    find_device()
    docker_run(file)
