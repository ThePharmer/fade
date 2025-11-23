# FADE Setup Guide

Complete walkthrough to run the definitive FADE test on your local machine.

## Requirements

- **GPU:** GTX 1080 or better (8GB+ VRAM)
- **CPU:** Intel i9 9900k or equivalent
- **Python:** 3.8+
- **OS:** Linux, macOS, or Windows with WSL2

## Step 1: Clone the Repository

```bash
git clone https://github.com/ThePharmer/fade.git
cd fade
```

Or if you already have it:
```bash
cd /path/to/fade
git pull origin main
```

## Step 2: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows
```

## Step 3: Install PyTorch with CUDA

For GTX 1080 (CUDA 11.8 or 12.1 recommended):

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is working:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce GTX 1080
```

## Step 4: Install Dependencies

```bash
pip install numpy tqdm
```

## Step 5: Verify Installation

```bash
# Quick sanity check
python -c "
import sys
sys.path.insert(0, 'src')
from fade.config import get_default_config
from fade.fade_model import FADEModel
config = get_default_config()
model = FADEModel(config)
print(f'Model parameters: {model.count_parameters():,}')
print('Setup OK!')
"
```

Expected output:
```
Model parameters: 536,704
Setup OK!
```

## Step 6: Run the Definitive Test

### Quick Test (~2 minutes)
Verify everything works:
```bash
python definitive_test.py --quick
```

### Full Test (~10-15 minutes)
Get statistically reliable results:
```bash
python definitive_test.py
```

### Custom Configuration
```bash
# More seeds for higher confidence
python definitive_test.py --seeds 10

# More epochs for stronger signal
python definitive_test.py --epochs 150

# Both
python definitive_test.py --seeds 10 --epochs 150
```

## Expected Output

The test runs 5 conditions across multiple random seeds:

| Condition | What it tests |
|-----------|---------------|
| `computed` | Full FADE with learned strength tracking |
| `random` | Same noise, but random targeting |
| `constant` | Uniform noise everywhere |
| `cal_loss_only` | Calibration loss without degradation |
| `baseline` | Standard training (no FADE) |

### Sample Output
```
======================================================================
AGGREGATED RESULTS
======================================================================

Method          ECE                  Fuzz-Err Corr        Accuracy
----------------------------------------------------------------------
computed        0.0423 +/- 0.0051    0.8234 +/- 0.0312    0.9150 +/- 0.0089
random          0.0478 +/- 0.0067    0.1523 +/- 0.0891    0.9134 +/- 0.0102
constant        0.0491 +/- 0.0073    -0.0891 +/- 0.1234   0.9128 +/- 0.0098
cal_loss_only   0.0445 +/- 0.0058    0.7891 +/- 0.0456    0.9142 +/- 0.0091
baseline        0.2891 +/- 0.0234    0.2134 +/- 0.1567    0.9156 +/- 0.0087
```

## Interpreting Results

### Key Comparisons

1. **computed vs baseline**
   - Large improvement → FADE works
   - Similar → FADE doesn't help

2. **computed vs random**
   - computed << random → Targeting matters (mechanism works!)
   - computed ≈ random → Just noise regularization

3. **computed vs cal_loss_only**
   - computed << cal_loss_only → Degradation helps
   - computed ≈ cal_loss_only → Calibration loss does most of the work

4. **Fuzz-Error Correlation**
   - computed >> random → Mechanism creates meaningful uncertainty signals
   - computed ≈ random → No meaningful difference

### Verdict Categories

| Verdict | Meaning |
|---------|---------|
| **STRONG SUCCESS** | FADE's targeting mechanism provides real calibration benefit |
| **PARTIAL SUCCESS** | Mechanism creates useful uncertainty signals, but ECE improvement is from regularization |
| **CALIBRATION LOSS DOMINATES** | The explicit loss term does most of the work |
| **INCONCLUSIVE** | Mixed results, may need longer training |

## Results File

Results are saved to `definitive_results.json`:
```json
{
  "config": {
    "epochs": 100,
    "seeds": 5,
    "d_model": 128,
    "n_layers": 2
  },
  "summary": {
    "computed": {"ece_mean": 0.0423, "ece_std": 0.0051, ...},
    ...
  },
  "raw_results": {...}
}
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
# Edit definitive_test.py line ~340
config.training.batch_size = 16  # Default is 32
```

### Slow Training
Ensure CUDA is being used:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall PyTorch with CUDA support.

### Import Errors
Make sure you're in the fade directory and venv is activated:
```bash
cd /path/to/fade
source venv/bin/activate
python definitive_test.py
```

## Next Steps

If results show promise (STRONG SUCCESS or PARTIAL SUCCESS with high correlation):
1. Consider implementing FADE on GPT-2 for real language modeling
2. Test on different datasets (longer sequences, harder tasks)
3. Explore using the fuzz-error correlation for retrieval triggering

If results show CALIBRATION LOSS DOMINATES:
1. The core insight (train model to predict its own errors) is valuable
2. The degradation mechanism may not be necessary
3. Consider simpler calibration approaches
