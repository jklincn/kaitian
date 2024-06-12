import argparse
import json
import os
import shutil
import subprocess

import docker

cuda_nums = 0
mlu_nums = 0
total_nums = 0


USE_CUDA = True if os.environ.get("USE_CUDA", 0) == "1" else False
USE_MLU = True if os.environ.get("USE_MLU", 0) == "1" else False
MASTER_ADDR = ""
MASTER_PORT = os.environ.get("MASTER_PORT", 12355)


def find_cuda():
    if shutil.which("nvidia-smi") is None:
        return
    try:
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
                    "ok"
                    if parts[2] == parts[4] and parts[3] == parts[5]
                    else "downgraded"
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
    except subprocess.CalledProcessError as e:
        print("nvidia-smi error: ", e)
        exit()
    except Exception as e:
        print("find_cuda error: ", e)
        exit()


def find_mlu():
    if shutil.which("cnmon") is None:
        return
    try:
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
    except subprocess.CalledProcessError as e:
        print("cnmon error: ", e)
        exit()
    except Exception as e:
        print("find_mlu error:", e)
        exit()


# NB: currently only test GTX1080 with Driver Version 520.61.05 (CUDA 11.8)
def docker_run_cuda(file):
    client = docker.from_env()
    image = "jklincn/kaitian:cuda-torch1.13.1-cuda11.6-cudnn8-devel"
    try:
        node_cuda = client.containers.get("node_cuda")
        node_cuda.remove(force=True)
    except docker.errors.NotFound:
        pass
    try:
        print("Running CUDA container...")
        node_cuda = client.containers.run(
            detach=True,
            network="kaitian",
            name="node_cuda",
            environment={
                "MASTER_ADDR": MASTER_ADDR,
                "MASTER_PORT": MASTER_PORT,
                "TOTAL_NUMS": total_nums,
            },
            device_requests=[
                docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
            ],
            shm_size="16G",
            volumes=[
                f"{os.path.abspath(file)}:/{os.path.basename(file)}",
                f"{os.path.dirname(os.path.abspath(file))}/data:/data",
            ],
            working_dir="/",
            image=image,
            command=f"mpirun -np 1 --allow-run-as-root python /{os.path.basename(file)}",
        )
        # read container output until container exit
        for line in node_cuda.logs(stream=True, follow=True):
            print(line.decode().strip(), flush=True)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Stop and remove all containers.")

    except Exception as e:
        print("docker_run_cuda error: ", e)
    finally:
        while True:
            try:
                node_cuda = client.containers.get("node_cuda")
                node_cuda.remove(force=True)
                return
            except docker.errors.NotFound:
                continue
            except UnboundLocalError:
                continue


# NB: currently only test MLU370
def docker_run_mlu(file):
    client = docker.from_env()
    image = "jklincn/kaitian:mlu-v1.17.0-torch1.13.1-ubuntu20.04-py310"
    try:
        node_mlu = client.containers.get("node_mlu")
        node_mlu.remove(force=True)
    except docker.errors.NotFound:
        pass
    try:
        device = ["/dev/cambricon_ctl"]
        for i in range(mlu_nums):
            device.extend([f"/dev/cambricon_dev{i}", f"/dev/cambricon_ipcm{i}"])
        print("Running MLU container...")
        node_mlu = client.containers.run(
            detach=True,
            network="kaitian",
            name="node_mlu",
            environment={
                "MASTER_ADDR": MASTER_ADDR,
                "MASTER_PORT": MASTER_PORT,
                # "MLU_NUMS": mlu_nums,
                "TOTAL_NUMS": total_nums,
            },
            devices=device,
            shm_size="16G",
            volumes=[
                f"{os.path.abspath(file)}:/{os.path.basename(file)}",
                f"{os.path.dirname(os.path.abspath(file))}/data:/data",
            ],
            working_dir="/",
            image=image,
            command=f"mpirun -np 1 --allow-run-as-root python /{os.path.basename(file)}",
        )
        # read container output until container exit
        for line in node_mlu.logs(stream=True, follow=True):
            print(line.decode().strip(), flush=True)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Stop and remove all containers.")
    except Exception as e:
        print("docker_run_mlu error: ", e)
    finally:
        while True:
            try:
                node_mlu = client.containers.get("node_mlu")
                node_mlu.remove(force=True)
                return
            except docker.errors.NotFound:
                continue
            except UnboundLocalError:
                continue


def check_cuda():
    client = docker.from_env()
    base_image = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel"
    try:
        client.images.get(base_image)
    except docker.errors.ImageNotFound:
        print(f"Image {base_image} does not exist.")
        print(f"You must `docker pull {image}` first.")
        exit()
    image = "jklincn/kaitian:cuda-torch1.13.1-cuda11.6-cudnn8-devel"
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        print(f"Image {image} does not exist.")
        print(f"Building {image}...")

        build_command = [
            "docker",
            "build",
            "-t",
            image,
            "--build-arg",
            f"IMAGE={base_image}",
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
                break
            if output:
                print(output.strip(), flush=True)

    print(f"Use CUDA Image: {image}")


def check_mlu():
    client = docker.from_env()
    base_image = (
        "yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310"
    )
    try:
        client.images.get(base_image)
    except docker.errors.ImageNotFound:
        print(f"Image {base_image} does not exist.")
        print(
            f"You must get {base_image} first. See https://sdk.cambricon.com/download?component_name=PyTorch"
        )
        exit()
    image = "jklincn/kaitian:mlu-v1.17.0-torch1.13.1-ubuntu20.04-py310"
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        print(f"Image {image} does not exist.")
        print(f"Building {image}...")

        build_command = [
            "docker",
            "build",
            "-t",
            image,
            "--build-arg",
            f"IMAGE={base_image}",
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
                break
            if output:
                print(output.strip(), flush=True)

    print(f"Use MLU Image: {image}")


def find_device():
    global cuda_nums
    global mlu_nums
    global total_nums
    print("-----------------------------")
    if USE_CUDA:
        cuda_nums = find_cuda()
        check_cuda()
    if USE_MLU:
        mlu_nums = find_mlu()
        check_mlu()
    total_nums = cuda_nums + mlu_nums


def docker_run(file):
    global MASTER_ADDR
    client = docker.from_env()
    try:
        try:
            network = client.networks.get("kaitian")
        except docker.errors.NotFound:
            network = client.networks.create("kaitian", driver="bridge")
        if cuda_nums > 0:
            MASTER_ADDR = "node_cuda"
            docker_run_cuda(file)
        if mlu_nums > 0:
            if MASTER_ADDR == "":
                MASTER_ADDR = "node_mlu"
            docker_run_mlu(file)
    except docker.errors.APIError as e:
        print("docker run error: ", e)
    finally:
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
