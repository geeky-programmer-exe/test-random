# **HW3P2 Handout (JAX / Flax NNX Edition)**

Welcome to **HW3P2: Automatic Speech Recognition with an Encoder-Decoder Transformer in JAX**!

In this assignment, you will build and train a state-of-the-art Transformer encoder-decoder for end-to-end speech recognition using **JAX**, **Flax NNX**, and **Optax**.

---

## 📁 Directory Structure

Your current working directory should have the following files:

```
.
├── README.md
├── HW3P2_Student_Starter_Notebook_F26.ipynb
├── config.yaml
├── hw3_data_subset/
├── hw3lib/
│   ├── data/
│   │   ├── asr_dataset.py
│   │   ├── tokenizer.py
│   │   └── tokenizer_jsons/
│   ├── decoding/
│   │   └── sequence_generator.py
│   ├── model/
│   │   ├── attention.py
│   │   ├── decoder_layers.py
│   │   ├── encoder_layers.py
│   │   ├── masks.py
│   │   ├── norms.py
│   │   ├── positional_encoding.py
│   │   ├── speech_embedding.py
│   │   ├── sublayers.py
│   │   └── transformers.py
│   ├── trainers/
│   │   ├── asr_trainer.py
│   │   └── base_trainer.py
│   └── utils/
│       ├── create_lr_scheduler.py
│       └── create_optimizer.py
├── tests/
│   ├── test_mask_padding.py
│   ├── test_mask_causal.py
│   ├── test_norms.py
│   ├── test_positional_encoding.py
│   ├── test_rope.py
│   ├── test_attention.py
│   ├── test_sublayer_selfattention.py
│   ├── test_sublayer_feedforward.py
│   ├── test_sublayer_crossattention.py
│   ├── test_encoderlayer_selfattention.py
│   ├── test_decoderlayer_crossattention.py
│   ├── test_transformer_encoder_decoder.py
│   ├── test_decoding.py
│   ├── test_dataset_asr.py
│   └── testing_framework.py
└── requirements.txt
```

---

## 🧪 Running Tests

Each module in `hw3lib/` corresponds to a unit test script in `tests/`:

```bash
# Example: testing RMSNorm
python tests/test_norms.py

# Example: testing MultiheadAttention
python tests/test_attention.py

# Example: testing full Encoder-Decoder Transformer
python tests/test_transformer_encoder_decoder.py
```

---

## Setup

This handout trains an encoder-decoder Transformer with **JAX**, **Flax NNX**, and **Optax**. Install the stack from `requirements.txt` on Local, Colab, and Kaggle. On PSC, use the shared class environment below and do not install packages yourself.

Follow the section for the environment you are using. The notebook's working directory must be this handout directory so `import hw3lib` resolves. Pass the local tests on `hw3_data_subset` before moving to a GPU runtime.

After setup, set `data.root` in `config.yaml` to the `hw3p2_data` folder for that environment.

### Local

Most of the implementation work can be done on a CPU with the provided `hw3_data_subset`. Pass the tests in `tests/` here before training on a GPU.

#### Step 1: Create a conda environment

```bash
# Deactivate any active environment first
conda create -n hw3p2-jax python=3.12.4
```

#### Step 2: Activate it

```bash
conda activate hw3p2-jax
```

#### Step 3: Install the JAX stack

```bash
pip install --no-cache-dir --ignore-installed -r requirements.txt
```

This installs JAX, Flax (including NNX), Optax, and the other packages listed in `requirements.txt`.

#### Step 4: Open the notebook from this directory

The starter notebook already lives in the handout. If you launch Jupyter from somewhere else, either move the notebook here or set the working directory in the notebook:

```python
import os
os.chdir("path/to/student_handout")
```

#### Step 5: Select the `hw3p2-jax` kernel

In the notebook, choose the environment you just created as the kernel. `!ls` should show at least:

```
.
├── README.md
├── HW3P2_Student_Starter_Notebook_F26.ipynb
├── config.yaml
├── requirements.txt
├── hw3lib/
├── tests/
└── hw3_data_subset/
```

For local runs, set `data.root` to `hw3_data_subset/hw3p2_data`.

### Colab

Use a GPU runtime (**Runtime → Change runtime type → GPU**) so JAX can see a CUDA device.

#### Step 1: Get the handout

Clone your repo. Create a classic personal access token under GitHub **Settings → Developer Settings → Personal Access Tokens**.

```python
import os

os.environ["GITHUB_TOKEN"] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "your_github_repo_name_here"
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}

# Later, from inside the repo, pull updates with:
# !cd {REPO_NAME} && git pull
```

#### Step 2: Install dependencies

```python
%pip install -r {REPO_NAME}/requirements.txt
```

Colab may ask you to restart the runtime after this install. Restart, then continue. Re-set `REPO_NAME` after a restart; it does not persist.

#### Step 3: Authenticate with Kaggle and download the data

Join the HW3P2 competition: https://www.kaggle.com/t/9a00ec2986e64408b581250f8a401880

```python
import os

os.environ["KAGGLE_USERNAME"] = "<your-username>"
os.environ["KAGGLE_API_TOKEN"] = "<your-key>"

import kaggle
api = kaggle.api
api.dataset_download_files(
    "cmu11785/hw-3-p-2-f-26-sandbox",
    path="/content/hw3_data",
    unzip=True,
)
```

#### Step 4: Move into the handout directory

```python
import os
REPO_NAME = "your_github_repo_name_here"  # re-set after every runtime restart
os.chdir(REPO_NAME)
!ls
```

Run this again after every runtime restart. `pwd` should be the handout directory.

Set `data.root` to `/content/hw3_data/hw3p2_data`.

### Kaggle

You can run the notebook on Kaggle, but this assignment is heavy and will usually be slower there than on Colab or PSC.

#### Step 1: Get the handout

```python
import os

os.environ["GITHUB_TOKEN"] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "your_github_repo_name_here"
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}
```

#### Step 2: Install the JAX stack

```python
!pip install -r {REPO_NAME}/requirements.txt
```

#### Step 3: Attach the dataset

Add the dataset as a notebook input. Do not download the zip into the notebook disk.

1. Open the Kaggle notebook.
2. Go to **Notebook → Input**.
3. Click **Add Input**.
4. Paste this URL: https://www.kaggle.com/datasets/cmu11785/hw-3-p-2-f-26-sandbox
5. Click **+** to attach it.

Kaggle mounts it under `/kaggle/input/`. Set `data.root` to the `hw3p2_data` directory inside that mount.

#### Step 4: Move into the handout directory

```python
import os
REPO_NAME = "your_github_repo_name_here"  # re-set after every runtime restart
os.chdir(REPO_NAME)
!ls
```

### PSC

PSC users share one Conda environment for this JAX assignment. Do not `pip install` on Bridges2.

Download the dataset onto the compute node's `$LOCAL` disk. That disk is fast and is wiped when the allocation ends, so download again each time you land on a new node. Stay on the same node and you can reuse the copy. See the [Bridges-2 file spaces](https://www.psc.edu/resources/bridges-2/user-guide#file-spaces) docs.

#### Step 1: Connect and open your Jet directory

1. In VS Code, install the **Remote - SSH** extension.
2. Open the command palette and run **Remote-SSH: Add New SSH Host**:

```bash
ssh <your_username>@bridges2.psc.edu
```

3. Connect, then **Explorer → Open Folder** and open:

```text
/jet/home/<your_username>
```

4. Upload the handout (or clone it) into that directory. The remaining terminal commands run in the VS Code integrated terminal.

#### Step 2: Request a GPU node

```bash
cd /jet/home/<your_username>
interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00 -A cis250019p
```

#### Step 3: Load Anaconda and activate the shared JAX environment

```bash
module load anaconda3
conda deactivate
conda activate /ocean/projects/cis250019p/mzhang23/TA/envs/IDLF26-JAX && export PYTHONNOUSERSITE=1
```

If the allocation drops the environment, run those three lines again.

#### Step 4: Start Jupyter and attach the kernel

```bash
jupyter notebook --no-browser --ip=0.0.0.0
```

In VS Code: **Kernel → Select Another Kernel → Existing Jupyter Server**, and paste the URL printed in the terminal:

```text
http://{hostname}:{port}/tree?token={token}
```

Example: `http://v011.ib.bridges2.psc.edu:8888/tree?token=e4b302434e68990f28bc2b4ae8d216eb87eecb7090526249`

#### Step 5: Clone the handout

From a notebook cell, with the working directory still `/jet/home/<your_username>`:

```python
import os

os.environ["GITHUB_TOKEN"] = "your-token"

GITHUB_USERNAME = "your-username"
REPO_NAME = "your_github_repo_name_here"
TOKEN = os.environ.get("GITHUB_TOKEN")
repo_url = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
!git clone {repo_url}
os.chdir(REPO_NAME)
!ls
```

#### Step 6: Authenticate with Kaggle and download the data to `$LOCAL`

Join the HW3P2 competition: https://www.kaggle.com/t/9a00ec2986e64408b581250f8a401880

```python
import os

os.environ["KAGGLE_USERNAME"] = "<your-username>"
os.environ["KAGGLE_API_TOKEN"] = "<your-key>"

import kaggle
api = kaggle.api

!mkdir -p $LOCAL/dataset
api.dataset_download_files(
    "cmu11785/hw-3-p-2-f-26-sandbox",
    path=f"{os.environ['LOCAL']}/dataset",
    unzip=True,
)
```

The features land at `/local/dataset/hw3p2_data`. Set `data.root` to that path.
