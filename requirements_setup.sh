#!/usr/bin/env bash
# requirements_setup.sh
# Run once on a fresh RunPod pod (PyTorch 2.4+, Python 3.11) to install all
# dependencies for the NTRS certified-density experiments.
#
# Usage:
#   scp requirements_setup.sh root@<pod-ip>:/workspace/
#   ssh root@<pod-ip> "bash /workspace/requirements_setup.sh"

set -euo pipefail

echo "============================================================"
echo "  NTRS experiment environment setup"
echo "  $(date)"
echo "============================================================"

# Set cache location FIRST before any HF calls
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p $HF_HOME

echo "============================================================"

# ── Verify PyTorch is already present (RunPod base image ships it) ────────────
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')"

# ── Pin transformers to 4.46.3 ────────────────────────────────────────────────
# Versions >= 4.47 introduce a continuous_batching module that imports
# torch.distributed.tensor.device_mesh, which is absent from PyTorch 2.4.1
# builds on RunPod. 4.46.3 is the last stable release before that change and
# supports GPT-2, Llama, Mistral, Qwen2 and full PEFT integration.
echo ""
echo "Installing Python packages..."
pip install --quiet --upgrade pip

apt update
apt install -y tmux

pip install --quiet \
    "transformers==4.46.3" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.3" \
    "peft>=0.12.0" \
    "datasets>=2.20.0" \
    "scipy>=1.13.0" \
    "matplotlib>=3.9.0" \
    "numpy>=1.26.0"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: Set HF_TOKEN in your environment before running this script (export HF_TOKEN=hf_...)." >&2
  exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

pip install hf_transfer -q

export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Smoke tests ───────────────────────────────────────────────────────────────
echo ""
echo "Running smoke tests..."

python3 - <<'EOF'
import sys

checks = [
    ("transformers GPT2LMHeadModel",
     "from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer"),
    ("transformers Conv1D",
     "from transformers.pytorch_utils import Conv1D"),
    ("peft",
     "from peft import get_peft_model, LoraConfig, TaskType"),
    ("datasets",
     "from datasets import load_dataset"),
    ("scipy.stats.beta",
     "from scipy.stats import beta"),
    ("matplotlib",
     "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt"),
    ("torch cuda",
     "import torch; assert torch.cuda.is_available(), 'CUDA not available'"),
]

all_ok = True
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  [OK]  {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}", file=sys.stderr)
        all_ok = False

if not all_ok:
    print("\n  One or more checks failed — see errors above.", file=sys.stderr)
    sys.exit(1)
else:
    print("\n  All checks passed. Environment is ready.")
EOF

echo ""
echo "============================================================"
echo "  Setup complete. You can now run:"
echo "    python3 lora_density_experiment.py --model gpt2 --lr 1e-4"
echo "    python3 basin_widening_experiment.py --model gpt2 --task sst2"
echo "============================================================"
