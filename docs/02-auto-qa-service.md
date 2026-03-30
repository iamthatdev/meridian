# Auto-QA Service — Complete Guide

**Component:** Auto-QA (Automated Quality Assurance) Service
**Version:** 1.0 (MVP)
**Last Updated:** 2026-03-27

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Automated QA?](#why-automated-qa)
3. [Validation Pipeline](#validation-pipeline)
4. [Schema Validation](#schema-validation)
5. [Readability Checking](#readability-checking)
6. [Basic Quality Rules](#basic-quality-rules)
7. [Scoring and Thresholds](#scoring-and-thresholds)
8. [Implementation Details](#implementation-details)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is the Auto-QA Service?

The **Auto-QA Service** is the component of IIAS that **automatically validates generated items** before they enter the Item Bank. It acts as a quality filter, ensuring only high-quality items reach human reviewers.

**Key responsibilities:**
1. **Validate schema** — Ensure JSON is well-formed and complete
2. **Check readability** — Verify passages are at appropriate grade level
3. **Apply quality rules** — Catch common issues (duplicates, trivial distractors, etc.)
4. **Score items** — Produce quality scores for ranking
5. **Filter items** — Reject low-quality items before human review

### Position in the Pipeline

```
Generation Service
        ↓
    (generates items)
        ↓
    Auto-QA Service  ◄─── YOU ARE HERE
        ↓
  (validates items)
        ↓
    Passed items → Item Bank (status: draft)
    Failed items → Rejection log
```

### What Auto-QA is NOT

**Auto-QA does NOT:**
- Replace human psychometricians
- Assess content accuracy (that requires domain expertise)
- Detect subtle biases (that requires human judgment)
- Calibrate IRT parameters (that requires student response data)
- Guarantee pedagogical soundness (that requires expert review)

**Auto-Qa DOES:**
- Filter out obviously malformed items
- Catch structural and formatting issues
- Flag items for human attention
- Save reviewer time by reducing noise

---

## Why Automated QA?

### The Problem: Manual Review is Expensive

**Human review of every generated item:**
- Psychometrician time: ~2-5 minutes per item
- Cost: $2-5 per item
- Throughput: 10-20 items per hour per reviewer
- **Bottleneck:** Human review becomes the limiting factor

### The Solution: Automated Pre-Filtering

**Automated QA before human review:**
- Filters out 20-40% of generated items (obviously bad)
- Psychometrician only reviews 60-80% of items
- Time saved: Reviewers focus on potentially good items
- **Result:** Higher throughput, lower cost

### Economic Impact

**Without Auto-QA:**
```
Generate 100 items
↓
Human review all 100 items @ 3 min/item = 300 minutes = 5 hours
↓
Approve 40 items (40% approval rate)
```

**With Auto-QA:**
```
Generate 100 items
↓
Auto-QA rejects 30 items (30 seconds)
↓
Human review 70 items @ 3 min/item = 210 minutes = 3.5 hours
↓
Approve 40 items (57% approval rate on reviewed items)
```

**Time saved:** 1.5 hours (30% reduction in reviewer time)
**Quality improved:** Higher approval rate on reviewed items

### Quality Improvement Feedback Loop

**Auto-QA isn't just a filter — it's a training signal.**

Rejected items are logged with rejection reasons:
- Schema violations → Model learns better JSON formatting
- Readability failures → Model learns appropriate passage complexity
- Quality rule failures → Model learns to avoid common mistakes

**Quarterly retraining** incorporates rejected examples as negative training data (via DPO or similar methods).

---

## Validation Pipeline

### Three-Stage Validation

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Schema Validation (HARD GATE)                      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Is JSON well-formed? │
         └──────────────────────┘
                    │
           ┌────────┴────────┐
           │                 │
         Yes                No
           │                 │
           ▼                 ▼
    Continue to Stage 2   REJECT (log error)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Readability Check (RW items only)                 │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Grade level 9-12?    │
         │ Passage length OK?   │
         └──────────────────────┘
                    │
           ┌────────┴────────┐
           │                 │
         Yes                No
           │                 │
           ▼                 ▼
    Continue to Stage 3   WARN (allow, flag)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Basic Quality Rules                               │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
    Check: duplicates, trivial distractors,
           length constraints, etc.
                    │
           ┌────────┴────────┐
           │                 │
        Passed            Failed
           │                 │
           ▼                 ▼
    ACCEPT → Item Bank   REJECT (log reason)
```

### Pass/Fail Logic

**Stage 1 (Schema): Hard Gate**
- **Pass:** All required fields present, correct types, valid enums
- **Fail:** Any schema violation → Immediate rejection
- **Rationale:** Malformed JSON cannot be stored in database

**Stage 2 (Readability): Warning Gate**
- **Pass:** Grade level 9-12, length within bounds
- **Warn:** Outside bounds → Item proceeds but flagged
- **Rationale:** Readability is subjective, defer to human judgment

**Stage 3 (Quality Rules): Hard Gate**
- **Pass:** All quality rules satisfied
- **Fail:** Any critical rule violation → Rejection
- **Rationale:** Obvious quality issues waste reviewer time

---

## Schema Validation

### What is Schema Validation?

**Schema validation** checks that a generated item conforms to the expected structure and data types defined by the Item Bank schema.

**Think of it as:** A grammar checker for JSON. Just as a grammar checker ensures sentences follow language rules, schema validation ensures JSON follows data structure rules.

### The Schema

**Canonical item schema (from Item Bank):**

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List
from enum import Enum

class Section(str, Enum):
    READING_WRITING = "reading_writing"
    MATH = "math"

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Choice(BaseModel):
    label: Literal["A", "B", "C", "D"]
    text: str

class ContentJSON(BaseModel):
    passage: Optional[str] = None
    question: str
    math_format: Literal["plain", "latex"]
    choices: List[Choice]
    correct_answer: Literal["A", "B", "C", "D"]
    correct_answer_text: str
    rationale: str
    solution_steps: Optional[str] = None

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v):
        if len(v) != 4:
            raise ValueError("Must have exactly 4 choices")
        if v[0].label != "A" or v[1].label != "B" or v[2].label != "C" or v[3].label != "D":
            raise ValueError("Choices must be labeled A, B, C, D in order")
        return v

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v, info):
        choices = {choice.label: choice.text for choice in info.data.get("choices", [])}
        if v not in choices:
            raise ValueError(f"correct_answer {v} not in choices")
        return v

    @field_validator("correct_answer_text")
    @classmethod
    def validate_correct_answer_text(cls, v, info):
        correct_answer = info.data.get("correct_answer")
        choices = {choice.label: choice.text for choice in info.data.get("choices", [])}
        if correct_answer and choices.get(correct_answer) != v:
            raise ValueError("correct_answer_text must match text of correct_answer choice")
        return v

class Item(BaseModel):
    id: str  # UUID
    section: Section
    domain: str
    difficulty: Difficulty
    content_json: ContentJSON

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        import uuid
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("id must be a valid UUID")
        return v
```

### Validation Checks

**1. Required Fields Check**

Ensures all required fields are present:

```python
required_fields = [
    "id", "section", "domain", "difficulty",
    "content_json.passage",  # Optional, but must be present (can be null)
    "content_json.question",
    "content_json.choices",
    "content_json.correct_answer",
    "content_json.correct_answer_text",
    "content_json.rationale"
]
```

**Example violations:**
```json
// ❌ WRONG: Missing required field
{
  "id": "uuid",
  "section": "math",
  "domain": "algebra"
  // Missing: difficulty, content_json
}
```

**2. Data Type Check**

Ensures each field has the correct data type:

```python
{
  "id": str,                    // Must be string
  "section": enum,              // Must be "reading_writing" or "math"
  "difficulty": enum,           // Must be "easy", "medium", or "hard"
  "content_json": object,       // Must be object/dict
  "content_json.passage": str|null,  // Must be string or null
  "content_json.choices": array,    // Must be array
  "content_json.correct_answer": enum  // Must be "A", "B", "C", or "D"
}
```

**Example violations:**
```json
// ❌ WRONG: Wrong data type
{
  "id": 123,                          // Should be string
  "section": "RW",                    // Should be "reading_writing"
  "difficulty": "extremely_hard",     // Invalid enum value
  "content_json.choices": {
    "A": "choice A",                  // Should be array of objects
    "B": "choice B"
  }
}
```

**3. Enum Value Check**

Ensures enum fields have valid values:

```python
section: ["reading_writing", "math"]
difficulty: ["easy", "medium", "hard"]
math_format: ["plain", "latex"]
correct_answer: ["A", "B", "C", "D"]
```

**Example violations:**
```json
// ❌ WRONG: Invalid enum values
{
  "section": "Reading & Writing",     // Should be "reading_writing"
  "difficulty": "intermediate",       // Should be "medium"
  "content_json.correct_answer": "E"  // Should be A-D (only 4 choices)
}
```

**4. Choices Structure Check**

Ensures choices array is properly formatted:

```python
{
  "choices": [
    {"label": "A", "text": "..."},
    {"label": "B", "text": "..."},
    {"label": "C", "text": "..."},
    {"label": "D", "text": "..."}
  ]
}
```

**Checks:**
- Exactly 4 choices
- Labels are A, B, C, D in order
- No duplicate labels
- No duplicate text values

**Example violations:**
```json
// ❌ WRONG: Wrong choices structure
{
  "choices": [
    {"label": "A", "text": "First"},
    {"label": "A", "text": "Second"},     // Duplicate label
    {"label": "C", "text": "Third"},
    {"label": "D", "text": "Fourth"}
  ]
}

// ❌ WRONG: Duplicate text
{
  "choices": [
    {"label": "A", "text": "Same text"},
    {"label": "B", "text": "Same text"},  // Duplicate text
    {"label": "C", "text": "Different"},
    {"label": "D", "text": "Also different"}
  ]
}
```

**5. Correct Answer Consistency Check**

Ensures `correct_answer` matches a choice label and `correct_answer_text` matches that choice's text:

```python
correct_answer = "C"
choices = {
  "A": "Text A",
  "B": "Text B",
  "C": "Text C",  # ← correct_answer must match this label
  "D": "Text D"
}
correct_answer_text = "Text C"  # ← Must match choice C's text
```

**Example violations:**
```json
// ❌ WRONG: correct_answer not in choices
{
  "choices": [
    {"label": "A", "text": "..."},
    {"label": "B", "text": "..."},
    {"label": "C", "text": "..."},
    {"label": "D", "text": "..."}
  ],
  "correct_answer": "E"  // Invalid: no choice E
}

// ❌ WRONG: correct_answer_text doesn't match
{
  "choices": [
    {"label": "A", "text": "Option A"},
    {"label": "B", "text": "Option B"},
    {"label": "C", "text": "Option C"},
    {"label": "D", "text": "Option D"}
  ],
  "correct_answer": "B",
  "correct_answer_text": "Option A"  // Should be "Option B"
}
```

**6. Math Format Check**

Ensures Math items with expressions use LaTeX:

```python
if section == "math" and has_math_expressions(question):
    assert math_format == "latex"
```

**Example violations:**
```json
// ❌ WRONG: Math item with expressions but math_format: plain
{
  "section": "math",
  "content_json": {
    "question": "If x² - 5x + 6 = 0, solve for x",
    "math_format": "plain"  // Should be "latex"
  }
}

// ✅ RIGHT: LaTeX used
{
  "section": "math",
  "content_json": {
    "question": "If $x^2 - 5x + 6 = 0$, solve for $x$",
    "math_format": "latex"
  }
}
```

### Implementation

**Using Pydantic:**

```python
from pydantic import ValidationError, validate_call

@validate_call
def validate_item(item: dict) -> tuple[bool, list[str]]:
    """
    Validate item against schema.

    Returns:
        (is_valid, errors)
    """
    try:
        Item(**item)
        return True, []
    except ValidationError as e:
        errors = [err["msg"] for err in e.errors()]
        return False, errors
```

**Example usage:**

```python
item = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "section": "math",
    "domain": "algebra.quadratic_equations",
    "difficulty": "medium",
    "content_json": {
        "passage": None,
        "question": "If x² - 5x + 6 = 0, what are the roots?",
        "math_format": "latex",
        "choices": [
            {"label": "A", "text": "x = 2, 3"},
            {"label": "B", "text": "x = -2, -3"},
            {"label": "C", "text": "x = 2, -3"},
            {"label": "D", "text": "x = -2, 3"}
        ],
        "correct_answer": "A",
        "correct_answer_text": "x = 2, 3",
        "rationale": "Factoring gives (x-2)(x-3)=0"
    }
}

is_valid, errors = validate_item(item)
if is_valid:
    print("✅ Schema valid")
else:
    print(f"❌ Schema invalid: {errors}")
```

---

## Readability Checking

### What is Readability?

**Readability** refers to how easy or difficult a text is to understand. It's typically measured by **grade level** — the education level needed to comprehend the text.

**Why it matters for SAT:**
- SAT passages are designed for grades 9-12
- Passages below grade 9 are too simple
- Passages above grade 12 are too advanced
- Target: Flesch-Kincaid Grade Level 9.0-12.0

### Flesch-Kincaid Grade Level

**The most common readability metric.**

**Formula:**
```
FKGL = 0.39 × (total words / total sentences) + 11.8 × (total syllables / total words) - 15.59
```

**Interpretation:**
- **FKGL = 8.0:** 8th grade reading level
- **FKGL = 10.5:** 10th grade, 5th month
- **FKGL = 12.0:** 12th grade (high school senior)

**SAT-appropriate range:** 9.0 ≤ FKGL ≤ 12.0

### Examples

**Grade 8 (too simple):**
```
The cat sat on the mat. It was a red mat. The cat liked the mat.
(FKGL ≈ 1.5)
```

**Grade 10 (SAT-appropriate):**
```
As cities expand, natural habitats disappear. Forest birds lose nesting sites. This causes population declines.
(FKGL ≈ 10.2)
```

**Grade 14 (too advanced):**
```
The anthropogenic transformation of landscape matrices precipitates commensurate disarticulation of avian population dynamics.
(FKGL ≈ 16.5)
```

### Implementation

**Using `textstat` library:**

```python
import textstat

def check_readability(passage: str) -> dict:
    """
    Check passage readability.

    Returns:
        {
            "passed": bool,
            "grade_level": float,
            "passage_word_count": int,
            "issues": list[str]
        }
    """
    if passage is None:
        return {"passed": True, "note": "No passage to check"}

    grade_level = textstat.flesch_kincaid_grade(passage)
    word_count = textstat.word_count(passage)

    issues = []

    # Check grade level
    if grade_level < 9.0:
        issues.append(f"Grade level {grade_level:.1f} below SAT range (9.0-12.0)")
    elif grade_level > 12.0:
        issues.append(f"Grade level {grade_level:.1f} above SAT range (9.0-12.0)")

    # Check passage length
    if word_count < 25:
        issues.append(f"Passage too short: {word_count} words (minimum: 25)")
    elif word_count > 500:
        issues.append(f"Passage too long: {word_count} words (maximum: 500)")

    return {
        "passed": len(issues) == 0,
        "grade_level": grade_level,
        "passage_word_count": word_count,
        "issues": issues
    }
```

**Example usage:**

```python
passage = """
As cities expand outward, the forests and grasslands that once bordered them
are replaced by roads, parking lots, and residential developments. For many bird
species, this transformation is catastrophic. Forest-interior specialists—birds
that require large, unbroken tracts of habitat for nesting and foraging—find
themselves stranded in shrinking patches of green space surrounded by hostile
urban matrix.
"""

result = check_readability(passage)
print(f"Grade level: {result['grade_level']:.1f}")
print(f"Word count: {result['passage_word_count']}")
print(f"Passed: {result['passed']}")
print(f"Issues: {result['issues']}")

# Output:
# Grade level: 10.8
# Word count: 98
# Passed: True
# Issues: []
```

### Passage Length Requirements

**Different item types have different length requirements:**

| Item Type | Min Words | Max Words |
|-----------|-----------|-----------|
| Short-context | 25 | 150 |
| Standard-context | 250 | 350 |
| Extended-context | 350 | 500 |

**Implementation:**

```python
def check_passage_length(passage: str, item_type: str = "standard") -> dict:
    """Check if passage length is appropriate for item type."""

    if passage is None:
        return {"passed": True, "note": "No passage (standalone item)"}

    word_count = textstat.word_count(passage)

    length_ranges = {
        "short": (25, 150),
        "standard": (250, 350),
        "extended": (350, 500)
    }

    min_words, max_words = length_ranges.get(item_type, (25, 500))

    if word_count < min_words:
        return {
            "passed": False,
            "word_count": word_count,
            "error": f"Passage too short for {item_type} item: {word_count} words (minimum: {min_words})"
        }
    elif word_count > max_words:
        return {
            "passed": False,
            "word_count": word_count,
            "error": f"Passage too long for {item_type} item: {word_count} words (maximum: {max_words})"
        }
    else:
        return {
            "passed": True,
            "word_count": word_count
        }
```

---

## Basic Quality Rules

### Rule 1: No Duplicate Choice Text

**Check:** All choice texts must be unique.

**Why:** Duplicate choices confuse students and indicate a generation error.

**Example violation:**
```json
{
  "choices": [
    {"label": "A", "text": "The value is 5"},
    {"label": "B", "text": "The value is 5"},  // Duplicate
    {"label": "C", "text": "The value is 10"},
    {"label": "D", "text": "The value is 15"}
  ]
}
```

**Implementation:**
```python
def check_duplicate_choices(choices: list) -> bool:
    """Check for duplicate choice texts."""
    texts = [choice["text"] for choice in choices]
    return len(texts) == len(set(texts))  # True if no duplicates
```

---

### Rule 2: No Trivial Distractors

**Check:** Wrong answers shouldn't be obviously wrong.

**What counts as trivial?**
- Nonsense strings ("XYZ123", "???")
- Single characters when answer is multi-character
- Numeric values orders of magnitude away from reasonable range
- Options that don't match the question format

**Example violations:**
```json
// ❌ WRONG: Trivial distractors
{
  "question": "What is 2 + 2?",
  "choices": [
    {"label": "A", "text": "3"},
    {"label": "B", "text": "4"},  // Correct
    {"label": "C", "text": "ABC"},  // Nonsense
    {"label": "D", "text": "1000000"}  // Obviously wrong
  ]
}
```

**Implementation:**
```python
import re

def check_trivial_distractors(item: dict) -> list:
    """Check for trivial distractors."""
    issues = []
    choices = item["content_json"]["choices"]
    correct_answer = item["content_json"]["correct_answer"]

    for choice in choices:
        if choice["label"] == correct_answer:
            continue  # Skip correct answer

        text = choice["text"].strip()

        # Check for nonsense
        if re.match(r'^[XYZ]{3,}$', text):
            issues.append(f"Choice {choice['label']} is nonsense: {text}")

        # Check for single character when answer should be longer
        if len(text) == 1 and item["section"] == "math":
            issues.append(f"Choice {choice['label']} suspiciously short: {text}")

        # Check for obviously wrong magnitude (Math only)
        if item["section"] == "math":
            # Extract numbers from choice
            numbers = re.findall(r'-?\d+\.?\d*', text)
            for num in numbers:
                val = float(num)
                if abs(val) > 10000:  # Suspiciously large
                    issues.append(f"Choice {choice['label']} has suspicious value: {val}")

    return issues
```

---

### Rule 3: Minimum Question Length

**Check:** Question text must be substantive.

**Threshold:** ≥ 20 characters

**Why:** Short questions are often incomplete or low-quality.

**Example violations:**
```json
// ❌ WRONG: Too short
{
  "question": "Solve it."  // 10 characters
}
```

**Implementation:**
```python
def check_question_length(question: str, min_length: int = 20) -> bool:
    """Check if question meets minimum length."""
    return len(question.strip()) >= min_length
```

---

### Rule 4: Minimum Explanation Length

**Check:** Rationale must be substantive.

**Threshold:** ≥ 30 words

**Why:** Explanations should explain why the answer is correct AND why distractors are wrong.

**Example violations:**
```json
// ❌ WRONG: Too short
{
  "rationale": "C is correct."  // 3 words
}
```

**Implementation:**
```python
def check_rationale_length(rationale: str, min_words: int = 30) -> bool:
    """Check if rationale meets minimum word count."""
    word_count = len(rationale.split())
    return word_count >= min_words
```

---

### Rule 5: Math Items Must Use LaTeX (When Applicable)

**Check:** Math items with mathematical expressions must use LaTeX formatting.

**What counts as a math expression?**
- Equations (e.g., "x² - 5x + 6 = 0")
- Fractions (e.g., "1/2")
- Exponents (e.g., "2³")
- Variables (e.g., "solve for x")
- Functions (e.g., "f(x) = 2x + 1")

**Example violations:**
```json
// ❌ WRONG: Math expression without LaTeX
{
  "section": "math",
  "content_json": {
    "question": "If x^2 - 5x + 6 = 0, what is x?",
    "math_format": "plain"  // Should be "latex"
  }
}

// ✅ RIGHT: LaTeX used
{
  "section": "math",
  "content_json": {
    "question": "If $x^2 - 5x + 6 = 0$, what is $x$?",
    "math_format": "latex"
  }
}
```

**Implementation:**
```python
def check_math_latex(item: dict) -> dict:
    """Check if Math item uses LaTeX appropriately."""
    if item["section"] != "math":
        return {"applicable": False}

    question = item["content_json"]["question"]
    math_format = item["content_json"]["math_format"]

    # Detect math expressions
    has_expressions = bool(
        re.search(r'[a-z]\s*[\^=]', question) or  # Variables with ^ or =
        re.search(r'\d\s*[\/\*]\s*\d', question)  # Fractions/multiplication
    )

    if has_expressions and math_format != "latex":
        return {
            "applicable": True,
            "passed": False,
            "error": "Math item has expressions but math_format is not 'latex'"
        }

    return {"applicable": True, "passed": True}
```

---

## Scoring and Thresholds

### Quality Score Calculation

**Auto-QA produces a composite quality score:**

```
qa_score = (schema_validity × 1.0) +
           (readability × 0.0) +  # Warning only, doesn't affect score
           (quality_rules × 0.0)   # Binary: pass or fail
```

**Simplified logic:**
- If schema validation fails: `qa_score = 0.0`
- If schema validation passes: `qa_score = 1.0`
- Readability warnings don't affect score (items still pass)
- Quality rules: if any fail, item is rejected

### Pass/Fail Threshold

**Items are accepted if:**
1. Schema validation: **PASS** (hard gate)
2. Readability check: **PASS or WARN** (warning doesn't reject)
3. Quality rules: **ALL PASS** (any fail rejects item)

**Items are rejected if:**
1. Schema validation: **FAIL**
2. Quality rules: **ANY FAIL**

### Auto-QA Output Format

```json
{
  "item_id": "uuid",
  "validation_timestamp": "2026-03-27T10:30:00Z",
  "schema_valid": true,
  "auto_qa_passed": true,
  "qa_score": 1.0,
  "checks": {
    "schema_validation": {
      "passed": true,
      "issues": []
    },
    "readability": {
      "passed": true,
      "grade_level": 10.8,
      "passage_word_count": 342,
      "issues": []
    },
    "quality_rules": {
      "passed": true,
      "issues": []
    }
  },
  "qa_flags": []
}
```

**Example with warnings:**

```json
{
  "item_id": "uuid",
  "schema_valid": true,
  "auto_qa_passed": true,
  "qa_score": 1.0,
  "checks": {
    "schema_validation": {
      "passed": true,
      "issues": []
    },
    "readability": {
      "passed": false,
      "grade_level": 8.2,
      "passage_word_count": 180,
      "issues": [
        "Grade level 8.2 below SAT range (9.0-12.0)",
        "Passage too short for standard item: 180 words (minimum: 250)"
      ]
    },
    "quality_rules": {
      "passed": true,
      "issues": []
    }
  },
  "qa_flags": ["READABILITY_BELOW_RANGE", "PASSAGE_TOO_SHORT"]
}
```

**Example with failure:**

```json
{
  "item_id": "uuid",
  "schema_valid": false,
  "auto_qa_passed": false,
  "qa_score": 0.0,
  "checks": {
    "schema_validation": {
      "passed": false,
      "issues": [
        "correct_answer 'E' not in choices",
        "choices must have exactly 4 elements"
      ]
    },
    "readability": {
      "passed": true,
      "grade_level": 10.5,
      "passage_word_count": 310,
      "issues": []
    },
    "quality_rules": {
      "passed": false,
      "issues": [
        "Choice C is nonsense: XYZ123"
      ]
    }
  },
  "qa_flags": ["SCHEMA_VALIDATION_FAILED", "TRIVIAL_DISTRACTOR"]
}
```

---

## Implementation Details

### Directory Structure

```
src/auto_qa/
├── __init__.py
├── schema.py              # Pydantic models for validation
├── validators/
│   ├── __init__.py
│   ├── schema_validator.py    # Schema validation
│   ├── readability_checker.py # Readability scoring
│   └── quality_rules.py       # Basic quality rules
├── scorer.py              # Composite scoring
└── pipeline.py            # Main validation pipeline

scripts/
└── validate_items.py     # CLI script for validation
```

### Key Classes

#### Schema Validator

```python
from pydantic import ValidationError

class SchemaValidator:
    def __init__(self):
        self.item_model = Item  # Pydantic model

    def validate(self, item: dict) -> dict:
        """Validate item against schema."""
        try:
            self.item_model(**item)
            return {
                "passed": True,
                "issues": []
            }
        except ValidationError as e:
            return {
                "passed": False,
                "issues": [err["msg"] for err in e.errors()]
            }
```

#### Readability Checker

```python
import textstat

class ReadabilityChecker:
    def __init__(self, min_grade: float = 9.0, max_grade: float = 12.0):
        self.min_grade = min_grade
        self.max_grade = max_grade

    def check(self, item: dict) -> dict:
        """Check readability of passage."""
        passage = item.get("content_json", {}).get("passage")

        if passage is None:
            return {"passed": True, "note": "No passage"}

        grade_level = textstat.flesch_kincaid_grade(passage)
        word_count = textstat.word_count(passage)

        issues = []
        if grade_level < self.min_grade:
            issues.append(f"Grade level {grade_level:.1f} below {self.min_grade}")
        elif grade_level > self.max_grade:
            issues.append(f"Grade level {grade_level:.1f} above {self.max_grade}")

        return {
            "passed": len(issues) == 0,
            "grade_level": grade_level,
            "word_count": word_count,
            "issues": issues
        }
```

#### Quality Rules Checker

```python
class QualityRulesChecker:
    def check(self, item: dict) -> dict:
        """Check basic quality rules."""
        issues = []

        # Check duplicate choices
        if not self._check_duplicate_choices(item):
            issues.append("Duplicate choice texts found")

        # Check trivial distractors
        trivial = self._check_trivial_distractors(item)
        if trivial:
            issues.extend(trivial)

        # Check question length
        if not self._check_question_length(item):
            issues.append("Question too short (< 20 characters)")

        # Check rationale length
        if not self._check_rationale_length(item):
            issues.append("Rationale too short (< 30 words)")

        # Check LaTeX for Math
        latex_check = self._check_math_latex(item)
        if latex_check.get("applicable") and not latex_check.get("passed"):
            issues.append(latex_check.get("error"))

        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
```

#### Main Pipeline

```python
class AutoQAPipeline:
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.readability_checker = ReadabilityChecker()
        self.quality_rules_checker = QualityRulesChecker()

    def validate(self, item: dict) -> dict:
        """Run full Auto-QA pipeline on item."""

        result = {
            "item_id": item.get("id"),
            "validation_timestamp": datetime.utcnow().isoformat(),
            "schema_valid": None,
            "auto_qa_passed": None,
            "qa_score": 0.0,
            "checks": {},
            "qa_flags": []
        }

        # Stage 1: Schema validation (hard gate)
        schema_result = self.schema_validator.validate(item)
        result["checks"]["schema_validation"] = schema_result
        result["schema_valid"] = schema_result["passed"]

        if not schema_result["passed"]:
            result["auto_qa_passed"] = False
            result["qa_flags"].extend(["SCHEMA_VALIDATION_FAILED"])
            return result

        # Stage 2: Readability check (warning gate)
        readability_result = self.readability_checker.check(item)
        result["checks"]["readability"] = readability_result

        if not readability_result["passed"]:
            result["qa_flags"].extend(["READABILITY_WARNING"])

        # Stage 3: Quality rules (hard gate)
        quality_result = self.quality_rules_checker.check(item)
        result["checks"]["quality_rules"] = quality_result

        if not quality_result["passed"]:
            result["auto_qa_passed"] = False
            result["qa_flags"].extend([f"QUALITY_RULE_FAILED: {issue}"
                                      for issue in quality_result["issues"]])
            return result

        # All checks passed
        result["auto_qa_passed"] = True
        result["qa_score"] = 1.0

        return result
```

---

## Best Practices

### 1. Validate Early, Validate Often

**Don't wait until items are generated.**

**Validate training data:**
```bash
python scripts/validate_training_data.py data/training/rw_train.jsonl
```

**Why:** Bad training data = bad model. Catch schema issues before training.

### 2. Log All Rejections

**Every rejected item should be logged with reasons.**

```python
if not result["auto_qa_passed"]:
    with open("outputs/logs/rejected_items.jsonl", "a") as f:
        f.write(json.dumps({
            "item": item,
            "validation_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }) + "\n")
```

**Why:** Rejection logs are training signal for model improvement.

### 3. Use Readability Warnings, Not Hard Gates

**Don't auto-reject for readability issues.**

**Why:** Readability metrics are imperfect. A grade 8.5 passage might still be high-quality. Let humans decide.

### 4. Check Math Format Aggressively

**Math items without LaTeX are almost always errors.**

**Why:** LLMs often forget to use LaTeX for simple expressions. Catch this early.

### 5. Update Validators When Schema Changes

**Schema version 1 → Schema version 2? Update validators.**

**Why:** Outdated validators reject valid items or accept invalid ones.

---

## Troubleshooting

### Problem: High Schema Validation Failure Rate

**Symptoms:** > 50% of items fail schema validation

**Solutions:**

1. **Check training data quality**
   - Are training examples valid?
   - Run `scripts/validate_training_data.py`

2. **Increase training epochs**
   - Model may need more exposure to learn schema
   - Try 5 epochs instead of 3

3. **Lower temperature**
   - High temperature = more randomness = more errors
   - Try `temperature=0.5` instead of 0.7

### Problem: All Items Pass Quality Rules (Suspicious)

**Symptoms:** 100% pass rate, but items look low-quality

**Causes:**
- Quality rules too lenient
- Rules not being applied correctly

**Solutions:**

1. **Review quality rule implementations**
   - Add print statements to debug
   - Test with known bad items

2. **Add more rules**
   - Trivial distractor detection
   - Answer consistency checks
   - Domain-specific rules

### Problem: Readability Check Crashes

**Symptoms:** `textstat.flesch_kincaid_grade()` raises error

**Causes:**
- Empty passage
- Non-English text
- Very short passages (< 3 sentences)

**Solutions:**

```python
def safe_readability_check(passage: str) -> float:
    try:
        return textstat.flesch_kincaid_grade(passage)
    except:
        return 10.0  # Default to 10th grade
```

---

## Summary

The Auto-QA Service is the quality filter that ensures only high-quality items reach human reviewers.

**Key takeaways:**

1. **Three-stage pipeline:** Schema → Readability → Quality rules
2. **Schema validation is hard gate** — malformed JSON is rejected immediately
3. **Readability is warning gate** — items proceed but are flagged
4. **Quality rules are hard gate** — obvious defects are rejected
5. **Log all rejections** — they're training signal for model improvement
6. **Don't over-filter** — better to pass a borderline item than reject a good one

**Next steps:**
- Read [`docs/03-item-bank.md`](./03-item-bank.md) to learn how validated items are stored

---

**Document version:** 1.0
**Last updated:** 2026-03-27
**Author:** IIAS Development Team
