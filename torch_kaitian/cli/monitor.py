import re
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def run_monitor(stop_flag):
    try:

        def get_cuda_utilization():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                cuda_data[i].append(int(line))

        def get_mlu_utilization():
            result = subprocess.run(["cnmon"], capture_output=True, text=True)
            lines = re.findall(r"\|\s\d+\s+/\s+.+?\|\s+(\d+)%\s+.*?\|", result.stdout)
            for i, line in enumerate(lines):
                mlu_data[i].append(int(line))

        cuda_data = [[] for _ in range(2)]
        mlu_data = [[] for _ in range(2)]

        while not stop_flag.value:
            start_time = time.time()
            get_cuda_utilization()
            get_mlu_utilization()
            elapsed = time.time() - start_time
            time.sleep(max(1 - elapsed, 0))

        plt.figure(figsize=(10, 6))

        # def moving_average(data, window_size):
        #     cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        #     ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        #     ma_vec = np.insert(ma_vec, 0, [ma_vec[0]] * (window_size - 1))
        #     return ma_vec

        # window_size = 50
        # # smoothed_single_losses = moving_average(single_losses, window_size)
        # smoothed_mlu_data = moving_average(mlu_data, window_size)

        for index, data in enumerate(cuda_data):
            plt.plot(data, label=f"GPU{index}")
            # Calculate and plot average utilization
            if data:
                avg_util = np.mean(data)
                plt.axhline(
                    y=avg_util,
                    color="r",
                    linestyle="--",
                    label=f"GPU{index} Average: {avg_util:.2f}%",
                )

        for index, data in enumerate(mlu_data):
            plt.plot(data, label=f"MLU{index}")
            # Calculate and plot average utilization
            if data:
                avg_util = np.mean(data)
                plt.axhline(
                    y=avg_util,
                    color="b",
                    linestyle="--",
                    label=f"MLU{index} Average: {avg_util:.2f}%",
                )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Utilization (%)")
        plt.title("Accelerators Utilization Over Time")
        plt.ylim(0, 100)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid(True)
        plt.savefig("utilization.png")
        plt.close()
    except KeyboardInterrupt:
        pass
