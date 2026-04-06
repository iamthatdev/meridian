# Create HuggingFace-based Training Script

## Task
Create a new training script `scripts/train_huggingface.py` that uses the standard HuggingFace Transformers library instead of the buggy custom training code.

## Requirements

### 1. Use Standard Libraries
- `transformers.Trainer` for training
- `transformers.TrainingArguments` for configuration
- `datasets.Dataset` for data loading
- Standard chat template formatting

### 2. Data Format
Our data is in JSONL format with this structure:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 3. Model Configuration
- Base model: `microsoft/phi-4` for math
- LoRA adapters: r=32, alpha=64, dropout=0.05
- 4-bit quantization (QLoRA)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 4. Training Configuration
- Learning rate: 2e-5
- Batch size: 4
- Gradient accumulation: 16
- Epochs: 3
- Max length: 2048 (math), 4096 (reading/writing)
- Warmup ratio: 0.05
- Logging: every 10 steps
- Save checkpoints: every 500 steps

### 5. Output Requirements
- Save checkpoints to `checkpoints/{section}/`
- Save training logs to `logs/training_{timestamp}.log`
- Print training progress
- Handle both `math` and `reading_writing` sections

### 6. Command Line Interface
```bash
python scripts/train_huggingface.py --section math --env production
```

## Key Features Needed

1. **Data Loading**: Load JSONL files and convert to HuggingFace Dataset
2. **Chat Formatting**: Use the model's chat_template to format messages
3. **Custom Trainer**: Handle special formatting requirements if needed
4. **Checkpointing**: Automatic saving and resuming
5. **Logging**: Both console and file logging
6. **Error Handling**: Clear error messages

## Dependencies Already Available
- transformers
- peft
- bitsandbytes
- datasets
- torch
- loguru

## What NOT to Do
- Do not use custom dataset classes (use standard HF datasets)
- Do not manually implement training loops (use Trainer)
- Do not create custom collators unless absolutely necessary
- Do not import from local `src.training` modules that have bugs

## Success Criteria
- Script runs without import errors
- Script loads data correctly
- Training starts and progresses
- Checkpoints are saved
- No UnboundLocalError, AttributeError, or data format errors

## Notes
- The existing `scripts/train_model.py` has multiple bugs - do not copy from it
- Start fresh with standard HuggingFace patterns
- Use PEFT/LoRA examples from HuggingFace documentation as reference
- Keep it simple and working rather than feature-rich
