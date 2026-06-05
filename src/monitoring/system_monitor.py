import time
import psutil
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pandas as pd

rows = []

for _ in range(30):

    rows.append(
        {
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent
        }
    )

    time.sleep(1)

df = pd.DataFrame(rows)

df.to_csv(
    "outputs/resource_monitoring.csv",
    index=False
)

if torch.cuda.is_available():

    print(
        torch.cuda.memory_allocated() / 1024**2,
        "MB"
    )

df = pd.read_csv(
    "outputs/resource_monitoring.csv"
)

plt.plot(df["cpu"])

plt.title("CPU Usage")
plt.xlabel("Time")
plt.ylabel("%")

plt.savefig(
    "outputs/cpu_usage.png"
)

print(df.describe())