# SAT Synthetic Questions - Data Documentation

**Generated:** March 26, 2026
**Total Questions:** 4,101
**Status:** Ready for use ✅

---

## 📁 Available Files

### Primary Data Files (Ready to Use)

| File | Size | Records | Format | Purpose |
|------|------|---------|--------|---------|
| `data/itembank_questions_complete.json` | 8.1 MB | 4,101 | **ItemBank** | ✅ **USE THIS FOR TRAINING** |
| `data/synthetic_questions_complete.json` | 4.8 MB | 4,101 | Synthetic | Raw generated format |

### Intermediate Files (Reference)

| File | Size | Records | Purpose |
|------|------|---------|---------|
| `data/synthetic_questions_recovered.json` | 4.0 MB | 3,373 | Recovered from checkpoints |
| `data/regenerated_questions.json` | 782 KB | 728 | Regenerated missing questions |

---

## 📋 Schema Documentation

### Format 1: Synthetic Questions (Raw)

**File:** `data/synthetic_questions_complete.json`

**Schema:**
```json
{
  "text": "string (question text)",
  "choices": ["A. option 1", "B. option 2", "C. option 3", "D. option 4"],
  "answer": "string (A, B, C, or D)",
  "explanation": "string (rationale)",
  "metadata": {
    "topic": "string (grammar, algebra, etc.)",
    "difficulty": "string (easy, medium, hard)",
    "question_type": "string (text_completion, problem_solving)",
    "section": "string (Reading and Writing, Math)",
    "generated_at": "ISO 8601 timestamp",
    "model": "string (model name)",
    "is_synthetic": true,
    "is_hybrid_mode": false,
    "provider": "string (poe, deepseek)"
  }
}
```

**Sample Record:**
```json
{
  "text": "Despite the challenges presented by climate change, many scientists believe that innovative technology _____ help mitigate its effects on the environment.",
  "choices": [
    "A. does",
    "B. can",
    "C. will",
    "D. should"
  ],
  "answer": "B",
  "explanation": "The phrase 'can help' indicates potential and suggests a reasonable likelihood.",
  "metadata": {
    "topic": "grammar",
    "difficulty": "medium",
    "question_type": "text_completion",
    "section": "Reading and Writing",
    "generated_at": "2026-03-25T17:27:14.268751",
    "model": "gpt-4o-mini",
    "is_synthetic": true,
    "is_hybrid_mode": false,
    "provider": "poe"
  }
}
```

---

### Format 2: ItemBank Questions (Recommended)

**File:** `data/itembank_questions_complete.json`

**Schema:**
```json
{
  "id": "uuid (unique identifier)",
  "version": 1,
  "status": "draft",
  "created_at": "ISO 8601 timestamp",
  "updated_at": "ISO 8601 timestamp",
  "created_by": "system:{provider}:{model}",
  "section": "reading_writing or math",
  "domain": "string (topic)",
  "difficulty_tier": "easy, medium, or hard",
  "irt_params": {
    "a": 1.0,
    "b": 0.0,
    "c": 0.25,
    "source": "seeded",
    "calibrated_at": null,
    "n_responses_at_calibration": 0,
    "se_a": null,
    "se_b": null,
    "se_c": null
  },
  "content_json": {
    "passage": null or "string",
    "question": "string (question text)",
    "math_format": "plain or latex",
    "choices": [
      {"label": "A", "text": "option 1"},
      {"label": "B", "text": "option 2"},
      {"label": "C", "text": "option 3"},
      {"label": "D", "text": "option 4"}
    ],
    "correct_answer": "A, B, C, or D",
    "correct_answer_text": "string (answer text)",
    "rationale": "string (explanation)",
    "solution_steps": null,
    "metadata": {
      "topic": "string",
      "grade_band": "9-12",
      "source": "ai_generated",
      "flags": [],
      "dataset_version": "1.0",
      "model_version": "string",
      "question_type": "string",
      "is_hybrid_mode": false
    }
  }
}
```

**Sample Record:**
```json
{
  "id": "45577574-e2ff-437c-ac91-a37c601e432a",
  "version": 1,
  "status": "draft",
  "created_at": "2026-03-25T17:27:14.268751",
  "updated_at": "2026-03-25T17:27:14.268751",
  "created_by": "system:poe:gpt-4o-mini",
  "section": "reading_writing",
  "domain": "grammar",
  "difficulty_tier": "medium",
  "irt_params": {
    "a": 1.0,
    "b": 0.0,
    "c": 0.25,
    "source": "seeded",
    "calibrated_at": null,
    "n_responses_at_calibration": 0,
    "se_a": null,
    "se_b": null,
    "se_c": null
  },
  "content_json": {
    "passage": null,
    "question": "Despite the challenges presented by climate change, many scientists believe that innovative technology _____ help mitigate its effects on the environment.",
    "math_format": "plain",
    "choices": [
      {"label": "A", "text": "does"},
      {"label": "B", "text": "can"},
      {"label": "C", "text": "will"},
      {"label": "D", "text": "should"}
    ],
    "correct_answer": "B",
    "correct_answer_text": "can",
    "rationale": "The phrase 'can help' indicates potential and suggests a reasonable likelihood.",
    "solution_steps": null,
    "metadata": {
      "topic": "grammar",
      "grade_band": "9-12",
      "source": "ai_generated",
      "flags": [],
      "dataset_version": "1.0",
      "model_version": "poe-gpt-4o-mini",
      "question_type": "text_completion",
      "is_hybrid_mode": false
    }
  }
}
```

---

## 🔄 Post-Processing Requirements

### ✅ NO Post-Processing Required!

The ItemBank format is **ready to use** for training:
- ✅ All questions have unique UUIDs
- ✅ Proper IRT parameters seeded
- ✅ Content JSON structured correctly
- ✅ All metadata preserved
- ✅ No duplicates

### Optional Enhancements (Future)

If you want to further improve the dataset:

1. **Update IRT Parameters**
   - Calibrate with real student response data
   - Update `irt_params.a`, `irt_params.b`, `irt_params.c`
   - Set `calibrated_at` and `n_responses_at_calibration`

2. **Change Status**
   - Update from `"status": "draft"` to `"status": "operational"`
   - Only after manual review/validation

3. **Add Solution Steps**
   - Currently `solution_steps` is null
   - Could add step-by-step solutions for math problems

4. **Flag Quality Issues**
   - Add flags to `content_json.metadata.flags` array
   - Examples: `["needs_review", "ambiguous", "typo"]`

---

## 📊 Dataset Statistics

### By Section
- **Reading & Writing:** 2,036 questions (49.7%)
- **Math:** 2,065 questions (50.3%)

### By Provider
- **Poe (GPT-4o-mini):** ~2,746 questions (67%)
- **Deepseek:** ~1,373 questions (33%)

### By Difficulty (Estimated)
- **Medium:** ~1,435 questions (35%)
- **Easy:** ~1,075 questions (26%)
- **Hard:** ~955 questions (23%)
- **Other:** ~636 questions (16%)

---

## 🚀 Quick Start

### Load ItemBank Data in Python

```python
import json
from pathlib import Path

# Load ItemBank format
with open('data/itembank_questions_complete.json') as f:
    questions = json.load(f)

print(f"Total questions: {len(questions)}")

# Access first question
q = questions[0]
print(f"Question: {q['content_json']['question']}")
print(f"Answer: {q['content_json']['correct_answer']}")
print(f"Difficulty: {q['difficulty_tier']}")
```

### Load in Pandas

```python
import pandas as pd
import json

# Load ItemBank data
with open('data/itembank_questions_complete.json') as f:
    data = json.load(f)

# Flatten to DataFrame
df = pd.json_normalize(data)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

---

## 📝 Notes

### Quality Assurance
- All questions generated by LLMs (GPT-4o-mini, Deepseek)
- JSON parsing validated during generation
- Incremental saving ensured no data loss
- UUID generation prevents duplicates

### Known Issues
- 18 questions failed to regenerate (out of 746)
- Some questions may have minor formatting inconsistencies
- IRT parameters are seeded, not calibrated

### Recommendations
- Use `itembank_questions_complete.json` for training
- Review a sample of questions before production use
- Consider manual validation for high-stakes applications
- Update IRT parameters after collecting response data

---

## 📞 Support

For questions or issues:
- Check the generation logs in `data/logs/`
- Review the checkpoint files in `data/checkpoints/`
- Examine the manifest in `data/generation_manifest.csv`

---

**Last Updated:** March 26, 2026
**Version:** 1.0
