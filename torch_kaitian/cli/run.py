import json
import os
import shutil
import subprocess
import docker
import threading

total_nums = 0
device_types = []
device_counts = {}
coordinator = ""
global_rank_start = 0

USE_CUDA = os.environ.get("USE_CUDA", "all")
USE_MLU = os.environ.get("USE_MLU", "all")

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
    device_counts["CUDA"] = len(gpu_info_list)
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
    device_counts["MLU"] = count
    return count


def check_image(device_type):
    match device_type:
        case "CUDA":
            base_image = CUDA_BASE_IMAGE
            image = CUDA_IMAGE
        case "MLU":
            base_image = MLU_BASE_IMAGE
            image = MLU_IMAGE
        case _:
            raise RuntimeError(
                f"[KaiTian][Error] Internal error: unmatched device_type {device_type}."
            )
    client = docker.from_env()
    try:
        client.images.get(base_image)
    except docker.errors.ImageNotFound:
        print(f"Image {base_image} does not exist.")
        match device_type:
            case "CUDA":
                print(f"You must `docker pull {base_image}` first.")
            case "MLU":
                print(
                    f"You must get {base_image} first. See https://sdk.cambricon.com/download?component_name=PyTorch"
                )
        exit()

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
            f"DEVICE={device_type}",
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
                    exit(f"Image build error: {exit_code}")
                break
            if output:
                print(output.strip(), flush=True)

    print(f"Use {device_type} Image: {image}")
    device_types.append(device_type)


def find_devices():
    print("-----------------------------")
    if USE_CUDA != "-1":
        find_cuda()
        if device_counts["CUDA"] > 0:
            check_image("CUDA")
            print("-----------------------------")
    if USE_MLU != "-1":
        find_mlu()
        if device_counts["MLU"] > 0:
            check_image("MLU")
            print("-----------------------------")


def run_container(device_type, file, gloo_rank, gloo_world_size, unknown_args):
    global global_rank_start
    print(f"Running {device_type} container...")
    client = docker.from_env()
    command = ["python", f"/{os.path.basename(file)}"] + unknown_args
    environment = {
        "KAITIAN_GLOBAL_RANK_START": global_rank_start,
        "KAITIAN_GLOBAL_WORLD_SIZE": total_nums,
        "KAITIAN_GLOO_RANK": gloo_rank,
        "KAITIAN_GLOO_WORLD_SIZE": gloo_world_size,
    }
    volumes = [
        "kaitian:/tmp/gloo",
        f"{os.path.abspath(file)}:/{os.path.basename(file)}",
        f"{os.path.dirname(os.path.abspath(file))}/data:/data",
        f"/home/lin/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints",
    ]
    match device_type:
        case "CUDA":
            try:
                if USE_CUDA == "all":
                    device_requests = [
                        docker.types.DeviceRequest(capabilities=[["gpu"]])
                    ]
                    global_rank_start += device_counts["CUDA"]
                else:
                    devices_ids = [
                        str(gpu_id) for gpu_id in map(int, USE_CUDA.split(","))
                    ]
                    device_requests = [
                        docker.types.DeviceRequest(
                            device_ids=devices_ids, capabilities=[["gpu"]]
                        )
                    ]
                    global_rank_start += len(devices_ids)
            except ValueError:
                exit(f"[KaiTian][Error] The provided USE_CUDA is invalid: {USE_CUDA}")
            devices = None
            image = CUDA_IMAGE
        case "MLU":
            device_requests = None
            # Compatible with MLU370
            devices = ["/dev/cambricon_ctl"]
            try:
                if USE_MLU == "all":
                    for i in range(device_counts["MLU"]):
                        devices.extend(
                            [f"/dev/cambricon_dev{i}", f"/dev/cambricon_ipcm{i}"]
                        )
                else:
                    devices_ids = list(map(int, USE_MLU.split(",")))
                    for i in range(device_counts["MLU"]):
                        if i in devices_ids:
                            devices.extend(
                                [f"/dev/cambricon_dev{i}", f"/dev/cambricon_ipcm{i}"]
                            )
            except ValueError:
                exit(f"[KaiTian][Error] The provided USE_MLU is invalid: {USE_MLU}")
            image = MLU_IMAGE
    return client.containers.run(
        detach=True,
        network="kaitian",
        name=device_type,
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
    logs = []
    for line in container.logs(stream=True, follow=True):
        logs.append(line.decode().strip())
        print(f"[{device_type}] {line.decode().strip()}", flush=True)
    with open(f"log_{device_type}.txt", "w") as log_file:
        log_file.write("\n".join(logs))


def docker_run(file, unknown_args):
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
        for gloo_rank, device_type in enumerate(device_types):
            containers[device_type] = run_container(
                device_type,
                file,
                gloo_rank,
                len(device_types),
                unknown_args,
            )
        # return
        print("--------- Output -----------")
        log_threads = []
        for device_type, container in containers.items():
            thread = threading.Thread(target=stream_logs, args=(container, device_type))
            thread.start()
            log_threads.append(thread)
        for thread in log_threads:
            thread.join()
        print("--------- Finish -----------")
        for device_type in device_types:
            print(
                f"{device_type} log file path: {os.path.dirname(os.path.abspath(__file__))}/log_{device_type}.txt",
                flush=True,
            )

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Stop and remove all containers.", flush=True)
    finally:
        # return
        all_containers = client.containers.list(all=True)
        for container in all_containers:
            container_name = container.attrs["Name"].strip("/")
            if container_name in device_types:
                container.remove(force=True)
        network.remove()
        volume.remove()


def use_devices():
    global total_nums
    try:
        if USE_CUDA == "-1":
            pass
        elif USE_CUDA == "all":
            total_nums += device_counts["CUDA"]
        else:
            devices_ids = [str(gpu_id) for gpu_id in map(int, USE_CUDA.split(","))]
            total_nums += len(devices_ids)
    except ValueError:
        exit(f"[KaiTian][Error] The provided USE_CUDA is invalid: {USE_CUDA}")
    try:
        if USE_MLU == "-1":
            pass
        elif USE_MLU == "all":
            total_nums += device_counts["MLU"]
        else:
            devices_ids = list(map(int, USE_MLU.split(",")))
            total_nums += len(devices_ids)
    except ValueError:
        exit(f"[KaiTian][Error] The provided USE_MLU is invalid: {USE_MLU}")


def run_kaitian(args, unknown_args):
    file = args.FILE
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found.")
    find_devices()
    use_devices()
    docker_run(file, unknown_args)
