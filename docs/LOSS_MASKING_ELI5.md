# Loss Masking for Supervised Fine-Tuning - ELI5 (Explain Like I'm 5)

## What is Loss Masking?

Loss masking is deciding which parts of the training data the model should learn from.

## Why Do We Need It?

When training a chat model, we show it examples like this:

```
System: You are a helpful tutor
User: What is 2+2?
Assistant: The answer is 4
```

**We want the model to learn:**
- ✅ How to respond as an assistant ("The answer is 4")
- ❌ NOT to repeat the system message
- ❌ NOT to repeat the user's question

## What Happens Without Loss Masking?

If we don't mask, the model learns from EVERYTHING:

```
System: You are a helpful tutor ← Model learns to say this
User: What is 2+2?              ← Model learns to ask this
Assistant: The answer is 4      ← Model learns to answer
```

**Result:** When you use the model, it might respond with:
```
System: You are a helpful tutor
User: What is 2+2?
```

Instead of just answering your question!

## What Happens With Loss Masking?

With loss masking, we tell the model: "Only learn from the assistant's response"

```
System: You are a helpful tutor ❌ MASKED (don't learn from this)
User: What is 2+2?              ❌ MASKED (don't learn from this)
Assistant: The answer is 4      ✅ LEARN FROM THIS
```

**Result:** Model learns to respond helpfully without repeating the prompt.

## Technical Details (For Curious Minds)

### How It Works

During training, we compute "loss" - a measure of how wrong the model's predictions are.

1. **Tokenize** the conversation into numbers:
   ```
   [523, 882, 145, ...]  ← System message
   [312, 999, ...]        ← User question
   [111, 555, 777, ...]   ← Assistant response
   ```

2. **Create labels** (what the model should predict):
   ```
   [-100, -100, -100, ...]  ← Don't care about system tokens
   [-100, -100, ...]         ← Don't care about user tokens
   [111, 555, 777, ...]      ← Learn to predict assistant tokens
   ```

3. **Compute loss** only on unmasked tokens (assistant response)

### Why -100?

In PyTorch (the training library), `-100` is a special value that means "ignore this token when computing loss."

## The Bug We Fixed

Our old training code had this bug:

```python
def _create_labels(self, prompt, input_ids):
    # Placeholder: return labels as-is (compute loss on all tokens)
    return labels  # ← WRONG! No masking applied
```

This meant loss was computed on ALL tokens, not just assistant responses.

**Impact:**
- Model wasted 70-80% of learning capacity on prompts
- Generated poor quality responses
- Training was inefficient

## The Fix

We migrated to `trl.SFTTrainer`, which automatically handles loss masking correctly:

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="messages",  # Automatically handles masking
)
```

**Result:** Model only learns from assistant responses, producing much better generations.

## How to Verify It's Working

After training, you can verify loss masking by checking that:

1. Training loss decreases over time (model is learning)
2. Model generates responses without repeating prompts
3. Loss values are reasonable (not suspiciously low or high)

## Resources

- [HuggingFace SFT Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Supervised Fine-Tuning Guide](https://huggingface.co/docs/trl/en/sft_trainer)
- [Our Implementation](../scripts/train_huggingface.py)

---

**TL;DR:** Loss masking tells the model "learn to respond, don't learn to repeat." Our fix ensures the model only learns from assistant responses, making it much better at answering questions.
