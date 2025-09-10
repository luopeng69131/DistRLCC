# DistRLCC: A Novel Distributed Reinforcement Learning Training System for Network Congestion Control

[📖 中文版 README](./README_CN.md)

**DistRLCC** is a **multi-machine distributed training system** for a reinforcement learning-based congestion control (CC) model.
This project implements a distributed training system for the [Indigo model](https://www.usenix.org/conference/atc18/presentation/yan-francis) in a **PyTorch RPC + Python3 environment** (the [original implementation](https://github.com/StanfordSNR/indigo) was based on Python 2.7 + TensorFlow).

<!-- ![System Architecture](./assets/pic.png) -->
<img src="./assets/pic.png" alt="System Architecture" width="600">

<!-- --- -->

## Requirements

* Operating System: **Ubuntu 18.04 or later**
* Python 3.7
* Dependencies:

  * [mahimahi](https://github.com/ravinet/mahimahi)
  * Python package dependencies (to be provided in `requirements.txt`)

Installation example:

```bash
sudo apt update
sudo apt install mahimahi -y

# Recommended: use a virtual environment
conda create -n DistRLCC python=3.7.9 -y
conda activate DistRLCC

# Install Python dependencies
pip install -r requirements.txt
```

<!-- --- -->

## Run

### Preparation

After reboot, enable IP forwarding (required for mahimahi simulation):

```bash
sudo sysctl -w net.ipv4.ip_forward=1
```

<!-- --- -->

### General Startup Rules

* **All machines** use the **same code and parameter format**. The only differences are:

  * `--node-num`: **Total number of machines in the cluster** (must be the same on all machines).
  * `--node-index`: **Index of the current machine** (master = `0`, workers = `1,2,3,...`).
* **All machines** must point `--IP` / `--port` to the **master node (node 0)**.
* The **total number of processes** during training is determined by the `NODES` configuration (see below). The master node spawns `2 + len(NODES[0])` processes; each worker node spawns `len(NODES[i])`.
* Use `--gpu -1` for CPU-only mode.
<!-- * Use `--tun` for VPN environments (OpenVPN, Tailscale). -->
* Use `--load` to resume training or load an existing model.

<!-- --- -->

### Command Template

> Replace `X` with your values:
> `NODE_NUM` = total number of machines;
> `NODE_INDEX` = index of this machine;
> `MASTER_IP` = master node’s IP;
> `PORT` defaults to 29513.

```bash
python main_mach_test_load.py \
  --gpu 0 \
  --node-index NODE_INDEX \
  --node-num NODE_NUM \
  --IP MASTER_IP \
  --port 29513

```

<!-- --- -->

### Example: 2 Machines (node 0 = master, node 1 = worker)

#### Master (node 0)

```bash
python main_mach_test_load.py \
  --gpu 0 \
  --node-index 0 \
  --node-num 2 \
  --IP 192.168.0.104 \
  --port 29513
```

#### Worker (node 1)

```bash
python main_mach_test_load.py \
  --gpu 0 \
  --node-index 1 \
  --node-num 2 \
  --IP 192.168.0.104 \
  --port 29513
```

<!-- --- -->

### Example: 3 Machines (node 0/1/2)

All three machines set `--node-num 3` and use the master IP (`192.168.0.104` in this example).

* Master (node 0):

```bash
python main_mach_test_load.py --gpu 0 --node-index 0 --node-num 3 --IP 192.168.0.104 --port 29513
```

* Worker (node 1):

```bash
python main_mach_test_load.py --gpu 0 --node-index 1 --node-num 3 --IP 192.168.0.104 --port 29513
```

* Worker (node 2):

```bash
python main_mach_test_load.py --gpu 0 --node-index 2 --node-num 3 --IP 192.168.0.104 --port 29513
```

<!-- --- -->

### Example: 4 Machines (node 0/1/2/3)

All four machines set `--node-num 4` and point to the master IP.

* Master (node 0):

```bash
python main_mach_test_load.py --gpu 0 --node-index 0 --node-num 4 --IP 192.168.0.104 --port 29513
```

* Worker (node 1):

```bash
python main_mach_test_load.py --gpu 0 --node-index 1 --node-num 4 --IP 192.168.0.104 --port 29513
```

* Worker (node 2):

```bash
python main_mach_test_load.py --gpu 0 --node-index 2 --node-num 4 --IP 192.168.0.104 --port 29513
```

* Worker (node 3):

```bash
python main_mach_test_load.py --gpu 0 --node-index 3 --node-num 4 --IP 192.168.0.104 --port 29513
```

<!-- --- -->

### `NODES` Configuration (Important)

`NODES` defines the number of **environment processes** on each machine. It is a list grouped by machine:

* `NODES[0]`: list of environment IDs on the master node (length = number of envs on master).
* `NODES[1]`: list of env IDs on worker node 1.
* `NODES[2]`: list of env IDs on worker node 2.
* …

> Example (adjust IDs and counts as needed):

```python
# a2c_ppo_acktr/config.py
NODES = [
  [0, 1, 2, 3],      # node 0 (master) has 4 envs
  [4, 5, 6],         # node 1 has 3 envs
  [7, 8],            # node 2 has 2 envs
  [9, 10, 11, 12],   # node 3 has 4 envs
]
```

* Master total processes = `2 + len(NODES[0])` (2 = TCP Server + Trainer).
* Worker i total processes = `len(NODES[i])`.
* **Total world\_size** = `sum(len(NODES[i]) for i in nodes) + 2` (already computed in code: `args.world_size = len(args.env_list) + 2`).
* If you add/remove machines, make sure to update:

  1. `--node-num` in the startup commands.
  2. `--node-index` for each machine.
  3. The `NODES` list length and assignments.

<!-- --- -->

## FAQ

* **Multi-machine connectivity**:

  * Option 1: Same LAN.
  * Option 2: VPN (e.g., [OpenVPN](https://openvpn.net/), [Tailscale](https://tailscale.com/)).
* **VPN scenario (machines is not in a LAN scenario)**: Use VPN such as Tailscale/OpenVPN and set `--IP` to the master’s VPN IP. (Maybe extra configuration to set TUN as the network interface in the code)
* **Ports & Firewall**: Ensure the master’s `--port` is reachable from all workers (training involves RPC and data traffic beyond just the main port).
* **IP Forwarding**: After reboot, always run:

  ```bash
  sudo sysctl -w net.ipv4.ip_forward=1
  ```
* **GPU/CPU mix**: Each machine can set its own `--gpu`, but homogeneous setups are recommended for simplicity.

<!-- --- -->

## Paper

For more details, please refer to the following paper:

```bibtex
@article{luo2023novel,
  title={A novel Congestion Control algorithm based on inverse reinforcement learning with parallel training},
  author={Luo, Pengcheng and Liu, Yuan and Wang, Zekun and Chu, Jian and Yang, Genke},
  journal={Computer Networks},
  volume={237},
  pages={110071},
  year={2023},
  publisher={Elsevier}
}
```

