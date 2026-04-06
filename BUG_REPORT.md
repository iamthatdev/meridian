# Bug Report: train_model.py Fixes Required

## Summary
The training script has multiple bugs that prevent it from running. These were discovered during RunPod deployment.

## Bugs to Fix

### Bug 1: Invalid File Structure (CRITICAL)
**File:** `scripts/train_model.py`
**Lines:** 1-3
**Problem:** Import statement appears before shebang
```python
from pathlib import Path      # ❌ Line 1: REMOVE THIS
#!/usr/bin/env python3         # Should be line 1
```
**Fix:** Delete line 1 entirely. Shebang `#!/usr/bin/env python3` must be on line 1.

---

### Bug 2: Import Statement in Docstring
**File:** `scripts/train_model.py`
**Line:** Around line 6-7
**Problem:** Import statement inside docstring
```python
"""
Training script for fine-tuning SAT item generation models.

from pathlib import Path        # ❌ REMOVE THIS LINE
Usage:
```
**Fix:** Remove the `from pathlib import Path` line from the docstring.

---

### Bug 3: Duplicate Import Inside train() Function
**File:** `scripts/train_model.py`
**Line:** Around line 202 (inside train() function)
**Problem:** Import shadows global import, causes UnboundLocalError
```python
def train(...):
    # Line 31 tries to use Path
    checkpoint_dir = Path(config.paths.checkpoint_dir) / section  # ❌ ERROR HERE

    # Line 202 imports Path inside function
    if resume_from:
        from pathlib import Path  # ❌ REMOVE THIS LINE
```
**Fix:** Remove the `from pathlib import Path` line from inside the train() function.

---

### Bug 4: Torch Import Inside train() Function
**File:** `scripts/train_model.py`
**Line:** Around line 202 (inside train() function)
**Problem:** Import shadows global import, causes UnboundLocalError
```python
def train(...):
    # Line 182 tries to use torch
    optimizer = torch.optim.AdamW(...)  # ❌ ERROR HERE

    # Line 202 imports torch inside function
    if resume_from:
        import torch  # ❌ REMOVE THIS LINE
```
**Fix:** Remove the `import torch` line from inside the train() function.

---

### Bug 5: Wrong Config Parameter Usage
**File:** `scripts/train_model.py`
**Line:** 192
**Problem:** References non-existent config attribute
```python
num_warmup_steps=config.training.warmup_steps,  # ❌ TrainingConfig has no warmup_steps
```
**Fix:** Calculate from warmup_ratio
```python
num_warmup_steps=int(num_training_steps * config.training.warmup_ratio),
```

---

### Bug 6: Incorrect Config Paths
**File:** `configs/production.yaml`
**Section:** paths:
**Problem:** Paths don't use `/root/meridian/` prefix
```yaml
paths:
  data_dir: /root/meridian/data  # ✅ Correct
  training_dir: /data/training   # ❌ Wrong - should be /root/meridian/data/training
  generated_dir: /data/generated # ❌ Wrong - should be /root/meridian/data/generated
  validated_dir: /data/validated # ❌ Wrong - should be /root/meridian/data/validated
  checkpoint_dir: /checkpoints   # ❌ Wrong - should be /root/meridian/checkpoints
  log_dir: /outputs/logs         # ❌ Wrong - should be /root/meridian/logs
```
**Fix:** All paths should be under `/root/meridian/`

---

## Files to Modify

1. **scripts/train_model.py** - Fix bugs 1-5
2. **configs/production.yaml** - Fix bug 6

## Testing

After fixes, verify:
```bash
python scripts/train_model.py --section math --env production --help
```

Should show help without errors.

## Notes

- The global imports at the top of the file are correct
- Only remove the duplicate imports INSIDE the train() function
- The TrainingConfig class uses `warmup_ratio` not `warmup_steps`
- All paths must be under `/root/meridian/` for RunPod deployment
