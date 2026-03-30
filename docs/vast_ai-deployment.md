# vast.ai Deployment Guide

Complete guide for training SAT item generation models on vast.ai GPU instances.

**Table of Contents:**
- [Phase 1: Pre-Rental Preparation](#phase-1-pre-rental-preparation-5-minutes)
- [Phase 2: Rent GPU Instance](#phase-2-rent-gpu-instance-on-vastaicom)
- [Phase 3: Connect and Setup Instance](#phase-3-connect-and-setup-instance-10-15-minutes)
- [Phase 4: Run Training](#phase-4-run-training-2-4-hours)
- [Phase 5: Collect Results and Shutdown](#phase-5-collect-results-and-shutdown-10-minutes)
- [Phase 6: Verify Locally](#phase-6-verify-locally-5-minutes)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Cost Estimation](#cost-estimation)
- [Quick Reference Commands](#quick-reference-commands)

---

## Phase 1: Pre-Rental Preparation (5 minutes)

Complete these steps **locally** before renting a GPU instance.

### Step 1: Get your HuggingFace token

```bash
# Go to: https://huggingface.co/settings/tokens
# Click "New token" → Generate → Copy it
# Store it securely (you'll need it on vast.ai)
```

**Why you need it:** The phi-4 model is gated and requires authentication to download.

### Step 2: Verify your code is ready

```bash
cd /Users/pradeep/projects/meridian

# Check data is linked correctly
wc -l data/splits/math_train.jsonl data/splits/readingwriting_train.jsonl
# Expected: 1755 math items, 1730 RW items

# Check production config exists
cat configs/production.yaml
# Should show Qwen 7B and phi-4 model IDs

# Verify training script exists
ls -la scripts/train_model.py
# Should exist and be executable
```

### Step 3: Create a .env.example for reference (optional)

```bash
cat > .env.vast << 'EOF'
APP_ENV=production
HF_TOKEN=your_huggingface_token_here
LOG_LEVEL=INFO
EOF
```

---

## Phase 2: Rent GPU Instance on vast.ai

### Step 4: Create vast.ai account (if needed)

```bash
# Go to: https://cloud.vast.ai/
# Sign up/login
# Add credits (usually $10-20 minimum)
```

### Step 5: Search for and rent instance

```bash
# On vast.ai website, use these search filters:
- **GPU:** A100 80GB OR A100 40GB
- **Min RAM:** 16GB
- **Min Disk:** 100GB
- **CUDA Version:** 12.x or 13.x
- **PyTorch:** Included (or search for "pytorch" in software)

# Recommended additional filters:
- Minimum CPU: 8 cores
- Minimum disk space: 100GB SSD
- Internet: Fast (100Mbps+)
- Direct SSH: Yes
- Disk IOPS: High (if available)

# Click "Rent" on your chosen instance
# Note the IP address and SSH command provided
```

**Expected cost:** ~$0.40-$0.80/hour
**Expected total cost:** $1-3 for full training run (3 epochs)

**Why A100:**
- phi-4 (14B) needs ~28GB VRAM at bf16, ~16GB at 4-bit
- Qwen 7B needs ~16GB VRAM at bf16, ~8GB at 4-bit
- 40GB is sufficient for 4-bit training, 80GB is comfortable for bf16

---

## Phase 3: Connect and Setup Instance (10-15 minutes)

### Step 6: SSH into the instance

```bash
# From your local terminal:
ssh -L 8080:localhost:8080 root@<IP-ADDRESS-PROVIDED-BY-VAST>

# Enter password when prompted (provided by vast.ai)
```

**Tip:** The `-L 8080` forwards localhost:8080 so you can access monitoring tools (tensorboard, etc.) if needed.

### Step 7: Update system and install Python

```bash
# Once logged into vast.ai instance:

# Update apt
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ if not present
sudo apt install python3.10 python3.10-venv python3-pip -y

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 8: Install PyTorch with CUDA support

```bash
# Install PyTorch (check https://pytorch.org/ for latest commands)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA works
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA available: True, Device: NVIDIA A100-SXM4-80GB
```

### Step 9: Install project dependencies

```bash
# Install required packages
pip install transformers>=4.30.0 peft>=0.5.0 trl>=0.7.0
pip install accelerate>=0.20.0 bitsandbytes>=0.41.0
pip install pydantic>=2.0.0 loguru>=0.7.0
pip install python-dotenv>=1.0.0
pip install psycopg2-binary>=2.9.0

# Install optional but recommended
pip install wandb tensorboard
```

### Step 10: Clone your repository

```bash
# Exit to your home directory first
cd ~

# OPTION A: Clone from GitHub (replace with your actual repo URL)
git clone <your-github-repo-url>
cd meridian

# OPTION B: Transfer from local machine (if repo is private)
# On your local machine (NOT vast.ai):
# scp -r /Users/pradeep/projects/meridian root@<IP-ADDRESS>:~/
```

### Step 11: Setup environment variables

```bash
# Create .env file
cat > .env << 'EOF'
APP_ENV=production
HF_TOKEN=your_actual_huggingface_token_here
LOG_LEVEL=INFO
EOF

# Verify it was created
cat .env
```

### Step 12: Verify data is accessible

```bash
cd ~/meridian

# Check data files exist
wc -l data/splits/math_train.jsonl
wc -l data/splits/readingwriting_train.jsonl

# Should show:
# 1755 data/splits/math_train.jsonl
# 1730 data/splits/readingwriting_train.jsonl
```

---

## Phase 4: Run Training (2-4 hours)

### Step 13: Start with a small test run (IMPORTANT!)

**Always test before committing to full training run:**

```bash
# Activate venv
source ~/venv/bin/activate
cd ~/meridian

# Run a tiny test first (validates everything works)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import load_config
import os

os.environ['APP_ENV'] = 'production'
config = load_config()

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    device_map='auto',
    load_in_4bit=True
)
print('✓ Model loaded successfully')
print('✓ Ready for training')
"

# Expected output: Model downloads (first time: ~2-3 min), then "✓ Ready for training"
```

### Step 14: Run actual training (Math section)

```bash
# Full training run for math
APP_ENV=production python scripts/train_model.py \
    --section math \
    --num_epochs 3

# This will:
# 1. Load Qwen 7B model (first time: ~2-3 min download)
# 2. Load training data (1,755 items)
# 3. Train for 3 epochs
# 4. Save checkpoints to outputs/checkpoints/math/
```

**Expected output:**
```
Loading model and tokenizer...
Training examples: 1755
Validation examples: 310
Epoch 1/3
Step 10/220: Loss: 2.3451
Step 20/220: Loss: 2.1234
...
Average training loss: 1.8765
Validation loss: 1.9234
Saved checkpoint: outputs/checkpoints/math/<timestamp>/epoch-1
...
Training complete!
```

**Duration:** 2-3 hours on A100 40GB, 1-2 hours on A100 80GB

### Step 15: Monitor training

```bash
# In another SSH terminal (leave training running in first):
ssh -L 6006:localhost:6006 root@<IP-ADDRESS>

# View logs in real-time
tail -f outputs/logs/train_$(date +%Y-%m-%d).log

# Or check GPU utilization
watch -n 1 nvidia-smi
# Look for GPU memory usage and utilization
# Should show ~40GB VRAM usage (A100 40GB) or ~70GB (A100 80GB)
```

---

## Phase 5: Collect Results and Shutdown (10 minutes)

### Step 16: Verify checkpoints were created

```bash
# List checkpoints
ls -lah outputs/checkpoints/math/

# Should show directories like:
# epoch-1/
# epoch-2/
# epoch-3/
# final/
# best/ (if validation used)

# Check checkpoint contents
ls outputs/checkpoints/math/epoch-1/
# Should see: adapter_config.json, adapter_model.safetensors, etc.
```

### Step 17: Download checkpoints to local machine

```bash
# On your LOCAL machine (not vast.ai):
cd /Users/pradeep/projects/meridian

# Create checkpoint directory
mkdir -p checkpoints/from_vast

# Download checkpoints via scp
scp -r root@<IP-ADDRESS>:/root/meridian/outputs/checkpoints/* checkpoints/from_vast/

# Or compress first on vast.ai to speed up transfer:
# On vast.ai:
cd ~/meridian
tar -czf checkpoints.tar.gz outputs/checkpoints/

# On local machine:
scp root@<IP-ADDRESS>:/root/meridian/checkpoints.tar.gz checkpoints/
tar -xzf checkpoints/checkpoints.tar.gz
```

### Step 18: Save training logs

```bash
# On vast.ai:
cd ~/meridian
cat outputs/logs/train_$(date +%Y-%m-%d).log > training_log.txt

# On local machine:
mkdir -p logs
scp root@<IP-ADDRESS>:/root/meridian/training_log.txt logs/vast_ai_math_$(date +%Y-%m-%d).log
```

### Step 19: Terminate instance

```bash
# On vast.ai website:
# 1. Go to "Your Instances"
# 2. Click "Stop" or "Destroy" on your instance
# 3. Confirm termination

# Important: This stops billing immediately!
```

---

## Phase 6: Verify Locally (5 minutes)

### Step 20: Test downloaded checkpoint

```bash
# Back on your local machine
cd /Users/pradeep/projects/meridian

# Verify checkpoint files exist
ls checkpoints/from_vast/

# Load and test checkpoint (optional)
python -c "
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    load_in_4bit=True
)
model = PeftModel.from_pretrained(model, 'checkpoints/from_vast/epoch-1')
print('✓ Checkpoint loads successfully')
"
```

---

## Troubleshooting Guide

### Issue: "CUDA out of memory"

**Symptoms:** Training crashes with OOM error, GPU memory full.

**Solution:** Reduce batch size in `configs/production.yaml`
```yaml
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 16  # Increase from 8
```

**Or use gradient checkpointing:**
```yaml
quantization:
  gradient_checkpointing: true  # Already enabled in config
```

### Issue: "Model download failed"

**Symptoms:** Error accessing HuggingFace model, authentication required.

**Solution:** Check HF_TOKEN is set correctly
```bash
echo $HF_TOKEN
# Should show your token, not empty

# Test manually:
python -c "from huggingface_hub import login; login(token='your_token')"
```

### Issue: "Data file not found"

**Symptoms:** Training fails with "Training file not found: data/splits/math_train.jsonl"

**Solution:** Verify symlinks exist
```bash
ls -la data/splits/
# Should show symlinks pointing to ../training/

# If missing, recreate:
cd data/splits
ln -sf ../training/math_train.jsonl math_train.jsonl
ln -sf ../training/math_val.jsonl math_val.jsonl
ln -sf ../training/rw_train.jsonl readingwriting_train.jsonl
ln -sf ../training/rw_val.jsonl readingwriting_val.jsonl
```

### Issue: "Training very slow"

**Symptoms:** GPU utilization is 0%, training not progressing.

**Check GPU is actually being used:**
```bash
nvidia-smi
# Should show GPU memory usage and utilization > 0%
```

**If GPU utilization is 0%:**
```bash
# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
# Should return: True

# Check device_map is working
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', device_map='auto'); print(m.hf_device_map)"
# Should show device map with cuda:0

# May need to manually set:
export CUDA_VISIBLE_DEVICES=0
```

### Issue: "ImportError: No module named 'src'"

**Symptoms:** Python can't find project modules.

**Solution:** Make sure you're in the meridian directory
```bash
cd ~/meridian
pwd
# Should show: /root/meridian

# Or add to PYTHONPATH
export PYTHONPATH=/root/meridian:$PYTHONPATH
```

### Issue: "Permission denied on .env"

**Symptoms:** Can't create .env file.

**Solution:** Check file permissions
```bash
ls -la .env  # Check if it exists and permissions

# Create with correct permissions
cat > .env << 'EOF'
APP_ENV=production
HF_TOKEN=your_token
EOF

# Or use different file name
export APP_ENV=production
export HF_TOKEN=your_token
```

### Issue: "SSH connection refused"

**Symptoms:** Can't connect to vast.ai instance.

**Solutions:**
1. Check IP address is correct
2. Check instance is actually running on vast.ai
3. Try without port forwarding first: `ssh root@<IP>`
4. Check your internet connection
5. If you're behind a firewall, try port 22 instead of default SSH port

### Issue: "Training loss is NaN"

**Symptoms:** Loss shows as NaN or infinity, model not learning.

**Solutions:**
1. Lower learning rate in config:
```yaml
training:
  learning_rate: 1e-5  # Reduce from 2e-5
```

2. Check gradient clipping:
```python
# In training script, look for:
# max_grad_norm=1.0
# Add if not present
```

3. Reduce batch size further:
```yaml
training:
  batch_size: 2
```

---

## Cost Estimation

| Instance Type | Hourly Cost | Expected Runtime | Total Cost per Section |
|---|---|---|---|
| A100 80GB | ~$0.80 | 1.5-2 hours | $1.20-$1.60 |
| A100 40GB | ~$0.40 | 2-3 hours | $0.80-$1.20 |

**Notes:**
- Costs are per section (Math OR Reading & Writing)
- Training both sections = 2x cost
- Runtime varies by data size and model complexity
- Add buffer time for model download and setup (~30 min)

**Cost optimization tips:**
- Use 40GB A100 with 4-bit quantization (config already uses this)
- Train during off-peak hours (some providers offer discounts)
- Stop instance immediately after training completes

---

## Quick Reference Commands

### SSH and Setup
```bash
# SSH with port forwarding
ssh -L 8080:localhost:8080 root@<IP-ADDRESS>

# Setup venv
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl accelerate bitsandbytes pydantic loguru python-dotenv
```

### Project Setup
```bash
# Clone repo
git clone <your-repo>
cd meridian

# Setup environment
echo "APP_ENV=production" > .env
echo "HF_TOKEN=<token>" >> .env
```

### Training Commands
```bash
# Test model loading
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    device_map='auto',
    load_in_4bit=True
)
print('✓ Ready')
"

# Train math section
APP_ENV=production python scripts/train_model.py --section math --num_epochs 3

# Train reading/writing section
APP_ENV=production python scripts/train_model.py --section reading_writing --num_epochs 3
```

### Monitoring
```bash
# View logs in real-time
tail -f outputs/logs/train_$(date +%Y-%m-%d).log

# Check GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check memory usage
free -h
```

### Download Results
```bash
# Download all checkpoints
scp -r root@<IP>:/root/meridian/outputs/checkpoints/* ./

# Compress first (faster for large files)
# On vast.ai:
tar -czf checkpoints.tar.gz outputs/checkpoints/

# On local:
scp root@<IP>:/root/meridian/checkpoints.tar.gz ./
tar -xzf checkpoints.tar.gz
```

---

## Post-Training: Next Steps

After you have checkpoints:

1. **Evaluate quality**
   ```bash
   python scripts/evaluate.py --section math --checkpoint outputs/checkpoints/math/latest
   ```

2. **Test inference**
   - Load checkpoint with PEFT
   - Generate sample items
   - Run through Auto-QA validation

3. **Iterate if needed**
   - If quality insufficient: train more epochs or adjust learning rate
   - If overfitting: add more data or regularization
   - If underfitting: train longer or increase model capacity

4. **Deploy for serving**
   - Use `scripts/deploy.py` to push checkpoint to inference endpoint
   - Or load checkpoint in FastAPI server for real-time generation

---

## Data Summary

**Available training data:**
- Math: 1,755 training + 310 validation = 2,065 items
- Reading & Writing: 1,730 training + 306 validation = 2,036 items
- **Total: 4,101 training items**

**Data format:** Chat messages with IIAS schema in assistant messages
**Quality:** Human-generated and curated items
**Source:** AI-generated with quality filtering

**Domain coverage:**
- Math: Algebra, Advanced Math, Problem Solving, Geometry & Trigonometry
- RW: Information & Ideas, Craft & Structure, Expression of Ideas, Standard English Conventions

---

## Configuration Notes

**Production models:**
- Reading & Writing: Qwen/Qwen2.5-7B-Instruct
- Math: microsoft/phi-4

**Training configuration:**
- LoRA r=32, alpha=64
- 4-bit quantization (NF4)
- Learning rate: 2e-5
- Batch size: 8
- Gradient accumulation: 8
- Epochs: 3

**Hardware requirements:**
- Minimum: A100 40GB (4-bit quantization)
- Recommended: A100 80GB (full precision or future model growth)
- RAM: 16GB+
- Disk: 100GB+

---

## FAQ

**Q: Can I use a smaller/cheaper GPU?**
A: Not recommended. The 14B phi-4 model needs at least 16GB VRAM even at 4-bit. RTX 3090 (24GB) might work but is much slower. A100 40GB is the cost-effective sweet spot.

**Q: Can I train both sections at once?**
A: No, train separately. Each section uses a different base model and has different optimal hyperparameters.

**Q: How do I know when training is working correctly?**
A: Watch for:
- Loss decreasing (not NaN or flat)
- Validation loss close to training loss (not much higher = no overfitting)
- GPU utilization > 80%
- Checkpoints being saved every epoch

**Q: What if training gets interrupted?**
A: The training script saves checkpoints after each epoch. You can resume from the last checkpoint by modifying the script or using the saved adapter weights.

**Q: Can I pause training and resume later?**
A: Yes, but you need to modify the training script to support resuming from checkpoints. Currently, it trains for the full num_epochs in one go.

---

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review logs in `outputs/logs/`
3. Check CLAUDE.md for project-specific configuration
4. Verify data files exist and are formatted correctly

**Common gotchas:**
- Forgot to set HF_TOKEN → Model download fails
- Wrong APP_ENV → Uses local config instead of production
- Data symlinks missing → "File not found" errors
- Insufficient disk space → Crash during checkpoint saving

**Always verify:**
1. Data exists before renting GPU
2. Test model loading before full training
3. Monitor first epoch closely
4. Download checkpoints before terminating instance
