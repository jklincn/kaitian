import argparse
import json
import os
import re
import shutil
import subprocess

import docker

cuda_nums = 0
mlu_nums = 0
total_nums = 0
hosts = []

USE_CUDA = True if os.environ.get("USE_CUDA", "1") == "1" else False
USE_MLU = True if os.environ.get("USE_MLU", "1") == "1" else False
MASTER_ADDR = ""
MASTER_PORT = os.environ.get("MASTER_PORT", 12355)

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
    hosts.append("node_cuda")


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
    hosts.append("node_mlu")


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


def run_container(host, file):
    client = docker.from_env()
    match host:
        case "node_cuda":
            print("Running CUDA container...")
            return client.containers.run(
                detach=True,
                network="kaitian",
                name="node_cuda",
                hostname="node_cuda",
                environment={
                    "MASTER_ADDR": MASTER_ADDR,
                    "MASTER_PORT": MASTER_PORT,
                    "TOTAL_NUMS": total_nums,
                },
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=["all"], capabilities=[["gpu"]]
                    )
                ],
                shm_size="16G",
                volumes=[
                    f"{os.path.abspath(file)}:/{os.path.basename(file)}",
                    f"{os.path.dirname(os.path.abspath(file))}/data:/data",
                ],
                working_dir="/",
                image=CUDA_IMAGE,
            )
        case "node_mlu":
            print("Running MLU container...")
            # Compatible with MLU370
            device = ["/dev/cambricon_ctl"]
            for i in range(mlu_nums):
                device.extend([f"/dev/cambricon_dev{i}", f"/dev/cambricon_ipcm{i}"])
            return client.containers.run(
                detach=True,
                network="kaitian",
                name="node_mlu",
                hostname="node_mlu",
                environment={
                    "MASTER_ADDR": MASTER_ADDR,
                    "MASTER_PORT": MASTER_PORT,
                    "TOTAL_NUMS": total_nums,
                },
                devices=device,
                shm_size="16G",
                volumes=[
                    f"{os.path.abspath(file)}:/{os.path.basename(file)}",
                    f"{os.path.dirname(os.path.abspath(file))}/data:/data",
                ],
                working_dir="/",
                image=MLU_IMAGE,
            )


def docker_run(file):
    global MASTER_ADDR
    client = docker.from_env()
    nodes = {}
    hosts_nums = len(hosts)
    # create network
    try:
        network = client.networks.get("kaitian")
    except docker.errors.NotFound:
        network = client.networks.create("kaitian", driver="bridge")
    # create container
    try:
        if hosts_nums > 0:
            MASTER_ADDR = hosts[0]
            for host in hosts:
                nodes[host] = run_container(host, file)
            # set nopassword login
            for host in hosts:
                pub_key_path = "/home/mpiuser/.ssh/id_ed25519.pub"
                authorized_keys_path = "/home/mpiuser/.ssh/authorized_keys"
                pub_key = nodes[host].exec_run(f"cat {pub_key_path}").output.decode()
                for other_host in hosts:
                    if host != other_host:
                        nodes[other_host].exec_run(
                            f"/bin/bash -c 'echo \"{pub_key.strip()}\" >> {authorized_keys_path}'"
                        )
            # run task
            execution = nodes[MASTER_ADDR].exec_run(
                cmd=f"mpirun -n {hosts_nums} --prefix /opt/openmpi --output tag-detailed,merge -host {','.join(hosts)} python /{os.path.basename(file)}",
                detach=False,
                stream=True,
            )
            print("--------- Output -----------")
            pattern = re.compile(r"\[\d+,\d+\]\[(.+?):\d+\]<stdout>:\s(.+)")
            max_width = max(len(host) for host in hosts)
            for line in execution.output:
                decoded_line = line.decode().strip()
                match = pattern.match(decoded_line)
                if match:
                    node_name = match.group(1)
                    message = match.group(2)
                    print(f"[{node_name:<{max_width}}] {message}", flush=True)
                else:
                    continue
        else:
            exit("No device available.")
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Stop and remove all containers.")
    finally:
        for host in hosts:
            try:
                node = client.containers.get(host)
                node.remove(force=True)
            except docker.errors.NotFound:
                continue
        network.remove()


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
