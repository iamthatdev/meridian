# Item Bank — Complete Guide

**Component:** Item Bank (PostgreSQL Database)
**Version:** 1.0 (MVP)
**Last Updated:** 2026-03-27

---

## Table of Contents

1. [Introduction](#introduction)
2. [Database Design](#database-design)
3. [Item Lifecycle](#item-lifecycle)
4. [Schema Definition](#schema-definition)
5. [Operations](#operations)
6. [Implementation Details](#implementation-details)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is the Item Bank?

The **Item Bank** is the **PostgreSQL database** that serves as the **single source of truth** for all SAT items generated and managed by IIAS. It stores:

- All generated items (draft, pretesting, operational, retired)
- Item metadata (section, domain, difficulty, IRT parameters)
- Auto-QA results (validation scores, flags)
- Human review decisions (approvals, rejections, edits)
- Calibration history (IRT parameter updates)
- Audit logs (every state change)

### Why PostgreSQL?

**PostgreSQL** was chosen for the Item Bank because:

1. **Relational integrity:** Enforces relationships between items, domains, reviews
2. **ACID compliance:** Guarantees data consistency (no partial updates)
3. **JSONB support:** Efficient storage and querying of JSON item content
4. **Full-text search:** Built-in text search for passages and questions
5. **Mature ecosystem:** Excellent Python libraries (psycopg2, SQLAlchemy)
6. **Scalability:** Handles millions of items with proper indexing

### Position in Architecture

```
Generation Service
        ↓
    (generates items)
        ↓
    Auto-QA Service
        ↓
  (validates items)
        ↓
    Item Bank ◄─── YOU ARE HERE
        ↓
   (stores items)
        ↓
    Review Workflow
        ↓
 (human approval)
        ↓
    Pretesting → Calibration → Operational
```

---

## Database Design

### High-Level Schema

```
┌─────────────────────────────────────────────────────────────┐
│                      Item Bank Tables                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ items            │  ← Main item records
├──────────────────┤
│ domains          │  ← Controlled vocabulary for domains
├──────────────────┤
│ calibration_log  │  ← IRT calibration history
├──────────────────┤
│ review_records   │  ← Human review decisions
├──────────────────┤
│ audit_log        │  ← All state changes
└──────────────────┘
```

### Entity Relationships

```
domains (1) ──< (many) items
                 │
                 │ (1)
                 │
                 (1)
           calibration_log
                 │
                 │ (many)
                 │
                 (1)
           review_records
                 │
                 │ (many)
                 │
                 (1)
           audit_log
```

---

## Item Lifecycle

### The Four States

Items move through **four lifecycle states**, each with a specific meaning and transition rules:

```
┌─────────┐    approve    ┌─────────────┐    calibrate    ┌─────────────┐
│  draft  │ ────────────▶ │ pretesting  │ ──────────────▶ │ operational │
└─────────┘               └─────────────┘                  └─────────────┘
    │                           │                              │
    │ reject                   │ retire                       │ retire
    │                          │                              │
    ▼                          ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              retired                              │
│                    (terminal state, never deleted)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### State Definitions

#### 1. draft

**Meaning:** Item has been generated and passed auto-QA, but not yet reviewed by a human.

**Characteristics:**
- **Safe to use:** NO — Cannot be administered to students under any circumstances
- **IRT parameters:** Seeded placeholders (irt_a=1.0, irt_b=0.0, irt_c=0.25, irt_source='seeded')
- **Visibility:** Visible to psychometricians in review queue
- **Count:** Most items in the Item Bank are in `draft` status

**Transition rules:**
- **Enter:** Created when Auto-QA passes an item
- **Exit:** Human approval → `pretesting`
- **Exit:** Human rejection → `retired`

**Example:**
```json
{
  "id": "uuid-123",
  "status": "draft",
  "created_at": "2026-03-27T10:00:00Z",
  "section": "math",
  "domain": "algebra.quadratic_equations",
  "auto_qa_passed": true,
  "qa_score": 0.95
}
```

---

#### 2. pretesting

**Meaning:** Item has been approved by a psychometrician and is eligible for field testing in **non-scored** positions.

**Characteristics:**
- **Safe to use:** YES — but only in non-scored positions (doesn't count toward student score)
- **IRT parameters:** Still seeded placeholders (not yet calibrated)
- **Purpose:** Collect response data for IRT calibration
- **Requirement:** N ≥ 500 responses before calibration

**Transition rules:**
- **Enter:** Human approval from `draft`
- **Exit:** Calibration successful → `operational`
- **Exit:** Calibration failed → `retired`
- **Exit:** Manual retirement → `retired`

**Example:**
```json
{
  "id": "uuid-123",
  "status": "pretesting",
  "created_at": "2026-03-27T10:00:00Z",
  "updated_at": "2026-03-28T14:30:00Z",  # Updated on approval
  "reviewed_at": "2026-03-28T14:30:00Z",
  "section": "math",
  "domain": "algebra.quadratic_equations",
  "irt_params": {
    "a": 1.0,
    "b": 0.0,
    "c": 0.25,
    "source": "seeded"
  }
}
```

---

#### 3. operational

**Meaning:** Item has been field tested, calibrated with real student response data, and is ready for use in **scored** test forms.

**Characteristics:**
- **Safe to use:** YES — Can be used in operational scored test forms
- **IRT parameters:** Empirically calibrated (irt_source='calibrated')
- **Quality gates:** All passed (fit statistics, SE thresholds, etc.)
- **Stability:** Parameters stable across calibration cycles

**Transition rules:**
- **Enter:** Successful calibration from `pretesting`
- **Exit:** Quality gate failure → `retired`
- **Exit:** Manual retirement → `retired`
- **Exit:** IRT parameter drift > 0.5 logits → `retired`

**Example:**
```json
{
  "id": "uuid-123",
  "status": "operational",
  "created_at": "2026-03-27T10:00:00Z",
  "updated_at": "2026-05-15T09:20:00Z",  # Updated on calibration
  "section": "math",
  "domain": "algebra.quadratic_equations",
  "irt_params": {
    "a": 1.15,
    "b": -0.3,
    "c": 0.19,
    "source": "calibrated",
    "calibrated_at": "2026-05-15T09:20:00Z",
    "n_responses_at_calibration": 523,
    "se_a": 0.12,
    "se_b": 0.08,
    "se_c": 0.03
  }
}
```

---

#### 4. retired

**Meaning:** Item is removed from use and will never be administered again.

**Characteristics:**
- **Safe to use:** NO — Never administered to students
- **Retention:** Permanently stored (never hard-deleted)
- **Reasons for retirement:**
  - Failed calibration (poor fit statistics)
  - Detected bias (DIF analysis)
  - IRT parameter drift
  - Psychometrician flag (content issue)
  - Obsolete content

**Transition rules:**
- **Enter:** From any state (rejection, failure, manual retirement)
- **Exit:** NEVER — terminal state

**Example:**
```json
{
  "id": "uuid-123",
  "status": "retired",
  "updated_at": "2026-06-01T10:00:00Z",
  "retirement_reason": "DIF detected: item performed differently across demographic groups",
  "irt_params": {
    "a": 0.72,  # Below 0.8 threshold
    "b": -0.25,
    "c": 0.28,
    "source": "calibrated"
  }
}
```

### State Transition Enforcement

**Database triggers enforce valid transitions:**

```sql
CREATE OR REPLACE FUNCTION enforce_item_status_transition()
RETURNS TRIGGER AS $$
BEGIN
    -- Draft → Pretesting
    IF OLD.status = 'draft' AND NEW.status = 'pretesting' THEN
        IF NOT EXISTS (SELECT 1 FROM review_records WHERE item_id = NEW.id AND decision IN ('approved', 'approved_with_edits')) THEN
            RAISE EXCEPTION 'Cannot transition to pretesting without human approval';
        END IF;
    END IF;

    -- Pretesting → Operational
    IF OLD.status = 'pretesting' AND NEW.status = 'operational' THEN
        IF NEW.irt_source != 'calibrated' THEN
            RAISE EXCEPTION 'Cannot transition to operational without calibrated IRT parameters';
        END IF;
        IF NEW.n_responses_at_calibration < 500 THEN
            RAISE EXCEPTION 'Cannot transition to operational with fewer than 500 responses';
        END IF;
    END IF;

    -- All other transitions are allowed (rejections, retirements)
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER item_status_transition_trigger
BEFORE UPDATE ON items
FOR EACH ROW
EXECUTE FUNCTION enforce_item_status_transition();
```

---

## Schema Definition

### Main Items Table

```sql
CREATE TABLE items (
    -- Primary key
    id UUID PRIMARY KEY,
    version INT DEFAULT 1,

    -- Lifecycle
    status VARCHAR(20) NOT NULL CHECK (status IN ('draft', 'pretesting', 'operational', 'retired')),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Section and domain
    section VARCHAR(20) NOT NULL CHECK (section IN ('reading_writing', 'math')),
    domain VARCHAR(100) NOT NULL REFERENCES domains(name),
    difficulty VARCHAR(10) NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),

    -- IRT parameters
    irt_a FLOAT NOT NULL DEFAULT 1.0,
    irt_b FLOAT NOT NULL DEFAULT 0.0,
    irt_c FLOAT NOT NULL DEFAULT 0.25,
    irt_source VARCHAR(20) NOT NULL DEFAULT 'seeded' CHECK (irt_source IN ('seeded', 'predicted', 'calibrated')),
    calibrated_at TIMESTAMP,
    n_responses_at_calibration INT DEFAULT 0,
    se_irt_a FLOAT,
    se_irt_b FLOAT,
    se_irt_c FLOAT,

    -- Item content (JSON)
    content_json JSONB NOT NULL,

    -- Auto-QA results
    auto_qa_passed BOOLEAN NOT NULL,
    qa_score FLOAT NOT NULL,
    qa_flags JSONB NOT NULL DEFAULT '[]',

    -- Model provenance
    model_version VARCHAR(100),

    -- Review metadata
    reviewed_at TIMESTAMP,
    reviewer_id UUID,

    -- Retirement (if applicable)
    retired_at TIMESTAMP,
    retirement_reason TEXT
);

-- Indexes for common queries
CREATE INDEX idx_items_status ON items(status);
CREATE INDEX idx_items_section ON items(section);
CREATE INDEX idx_items_domain ON items(domain);
CREATE INDEX idx_items_difficulty ON items(difficulty);
CREATE INDEX idx_items_section_domain_status ON items(section, domain, status);
CREATE INDEX idx_items_model_version ON items(model_version);

-- Full-text search on passage and question
CREATE INDEX idx_items_content_search ON items USING gin(to_tsvector('english',
    COALESCE(content_json->>'passage', '') || ' ' || COALESCE(content_json->>'question', '')
));
```

### Domains Table

```sql
CREATE TABLE domains (
    name VARCHAR(100) PRIMARY KEY,
    section VARCHAR(20) NOT NULL,
    category VARCHAR(100),
    description TEXT,
    target_percentage FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Reading & Writing domains
INSERT INTO domains (name, section, category, target_percentage) VALUES
('information_and_ideas.central_ideas_and_details', 'reading_writing', 'Information & Ideas', 6.25),
('information_and_ideas.command_of_evidence_textual', 'reading_writing', 'Information & Ideas', 6.25),
('information_and_ideas.inferences', 'reading_writing', 'Information & Ideas', 6.25),
-- ... (all 16 RW domains)

-- Math domains
INSERT INTO domains (name, section, category, target_percentage) VALUES
('algebra.linear_equations_one_variable', 'math', 'Algebra', 8.75),
('algebra.linear_equations_two_variables', 'math', 'Algebra', 8.75),
('algebra.linear_functions', 'math', 'Algebra', 8.75),
-- ... (all 16 Math domains)
```

### Calibration Log Table

```sql
CREATE TABLE calibration_log (
    log_id UUID PRIMARY KEY,
    item_id UUID NOT NULL REFERENCES items(id),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Calibration method
    calibration_engine_version VARCHAR(100),
    estimation_method VARCHAR(20) CHECK (estimation_method IN ('MAP', 'MMLE-EM')),

    -- Prior distributions
    prior_a_mean FLOAT,
    prior_a_sigma FLOAT,
    prior_b_mean FLOAT,
    prior_b_sigma FLOAT,
    prior_c_alpha FLOAT,
    prior_c_beta FLOAT,

    -- Anchor items used for equating
    anchor_items_used UUID[],

    -- Old parameters (before calibration)
    old_irt_a FLOAT,
    old_irt_b FLOAT,
    old_irt_c FLOAT,

    -- New parameters (after calibration)
    new_irt_a FLOAT,
    new_irt_b FLOAT,
    new_irt_c FLOAT,

    -- Standard errors
    se_irt_a FLOAT,
    se_irt_b FLOAT,
    se_irt_c FLOAT,

    -- Sample size
    n_responses INT NOT NULL,

    -- Fit statistics
    fit_s_x2_p_value FLOAT,
    fit_rmsea FLOAT,
    fit_infit_mnsq FLOAT,
    fit_outfit_mnsq FLOAT,

    -- Quality gate
    fit_passed BOOLEAN NOT NULL,

    -- Flags
    flags JSONB DEFAULT '[]',

    -- Equating run ID
    equating_run_id UUID
);

CREATE INDEX idx_calibration_log_item_id ON calibration_log(item_id);
CREATE INDEX idx_calibration_log_timestamp ON calibration_log(timestamp);
```

### Review Records Table

```sql
CREATE TABLE review_records (
    record_id UUID PRIMARY KEY,
    item_id UUID NOT NULL REFERENCES items(id),
    reviewer_id UUID NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Decision
    decision VARCHAR(30) NOT NULL CHECK (decision IN ('approved', 'approved_with_edits', 'rejected', 'escalated')),

    -- Rejection reasons (if rejected)
    rejection_reasons JSONB,

    -- Edits (if approved_with_edits)
    edits_made JSONB,

    -- Notes
    notes TEXT,

    -- Time spent reviewing
    review_duration_seconds INT
);

CREATE INDEX idx_review_records_item_id ON review_records(item_id);
CREATE INDEX idx_review_records_reviewer_id ON review_records(reviewer_id);
CREATE INDEX idx_review_records_decision ON review_records(decision);
```

### Audit Log Table

```sql
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY,
    item_id UUID NOT NULL REFERENCES items(id),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Action
    action_type VARCHAR(50) NOT NULL,  -- created, updated, approved, rejected, calibrated, retired
    actor_type VARCHAR(20) NOT NULL CHECK (actor_type IN ('user', 'system')),
    actor_id UUID,

    -- State change
    before_status VARCHAR(20),
    after_status VARCHAR(20),

    -- Change details
    changes JSONB,

    -- Request ID (for tracing)
    request_id UUID
);

CREATE INDEX idx_audit_log_item_id ON audit_log(item_id);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_log_action_type ON audit_log(action_type);
```

---

## Operations

### Creating Items

**When an item passes Auto-QA, it's inserted into the Item Bank.**

```python
import uuid
from datetime import datetime
import psycopg2
from psycopg2.extras import Json

def create_item(item: dict, qa_result: dict) -> str:
    """
    Create a new item in the Item Bank.

    Args:
        item: Generated item (from Generation Service)
        qa_result: Auto-QA validation result

    Returns:
        item_id (UUID)
    """
    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    item_id = str(uuid.uuid4())

    cur.execute("""
        INSERT INTO items (
            id, status, created_at, updated_at,
            section, domain, difficulty,
            irt_a, irt_b, irt_c, irt_source,
            content_json,
            auto_qa_passed, qa_score, qa_flags,
            model_version
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s,
            %s, %s, %s,
            %s
        )
    """, (
        item_id,
        'draft',  # All new items start as draft
        datetime.utcnow(),
        datetime.utcnow(),
        item['section'],
        item['domain'],
        item['difficulty'],
        1.0, 0.0, 0.25, 'seeded',  # Placeholder IRT parameters
        Json(item['content_json']),
        qa_result['auto_qa_passed'],
        qa_result['qa_score'],
        Json(qa_result['qa_flags']),
        item.get('model_version', 'unknown')
    ))

    # Log to audit
    cur.execute("""
        INSERT INTO audit_log (log_id, item_id, timestamp, action_type, actor_type, after_status, changes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        item_id,
        datetime.utcnow(),
        'created',
        'system',
        'draft',
        Json({'item': item, 'qa_result': qa_result})
    ))

    conn.commit()
    cur.close()
    conn.close()

    return item_id
```

---

### Querying Items

**Query draft items for review:**

```python
def get_draft_items(section: str = None, domain: str = None, limit: int = 50):
    """Query draft items for human review."""

    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    query = """
        SELECT id, section, domain, difficulty, content_json, qa_score, qa_flags
        FROM items
        WHERE status = 'draft'
    """
    params = []

    if section:
        query += " AND section = %s"
        params.append(section)

    if domain:
        query += " AND domain = %s"
        params.append(domain)

    query += " ORDER BY created_at ASC LIMIT %s"
    params.append(limit)

    cur.execute(query, params)
    items = cur.fetchall()

    cur.close()
    conn.close()

    return items
```

**Query operational items for export:**

```python
def get_operational_items(section: str, domain: str = None):
    """Query operational items for test assembly."""

    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    query = """
        SELECT id, section, domain, difficulty, content_json, irt_a, irt_b, irt_c
        FROM items
        WHERE status = 'operational'
        AND section = %s
    """
    params = [section]

    if domain:
        query += " AND domain = %s"
        params.append(domain)

    query += " ORDER BY domain, difficulty, irt_b"  # Sorted by difficulty

    cur.execute(query, params)
    items = cur.fetchall()

    cur.close()
    conn.close()

    return items
```

---

### Updating Items

**Approve an item (draft → pretesting):**

```python
def approve_item(item_id: str, reviewer_id: str):
    """Approve an item for pretesting."""

    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    # Update item status
    cur.execute("""
        UPDATE items
        SET status = 'pretesting',
            updated_at = %s,
            reviewed_at = %s,
            reviewer_id = %s
        WHERE id = %s
    """, (datetime.utcnow(), datetime.utcnow(), reviewer_id, item_id))

    # Create review record
    cur.execute("""
        INSERT INTO review_records (record_id, item_id, reviewer_id, timestamp, decision)
        VALUES (%s, %s, %s, %s, %s)
    """, (str(uuid.uuid4()), item_id, reviewer_id, datetime.utcnow(), 'approved'))

    # Log to audit
    cur.execute("""
        INSERT INTO audit_log (log_id, item_id, timestamp, action_type, actor_type, actor_id, before_status, after_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        item_id,
        datetime.utcnow(),
        'approved',
        'user',
        reviewer_id,
        'draft',
        'pretesting'
    ))

    conn.commit()
    cur.close()
    conn.close()
```

**Reject an item (any status → retired):**

```python
def reject_item(item_id: str, reviewer_id: str, rejection_reasons: list, notes: str = None):
    """Reject an item."""

    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    # Get current status
    cur.execute("SELECT status FROM items WHERE id = %s", (item_id,))
    old_status = cur.fetchone()[0]

    # Update item status
    cur.execute("""
        UPDATE items
        SET status = 'retired',
            updated_at = %s,
            retired_at = %s,
            retirement_reason = %s
        WHERE id = %s
    """, (datetime.utcnow(), datetime.utcnow(), '; '.join(rejection_reasons), item_id))

    # Create review record
    cur.execute("""
        INSERT INTO review_records (record_id, item_id, reviewer_id, timestamp, decision, rejection_reasons, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        item_id,
        reviewer_id,
        datetime.utcnow(),
        'rejected',
        Json(rejection_reasons),
        notes
    ))

    # Log to audit
    cur.execute("""
        INSERT INTO audit_log (log_id, item_id, timestamp, action_type, actor_type, actor_id, before_status, after_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        item_id,
        datetime.utcnow(),
        'rejected',
        'user',
        reviewer_id,
        old_status,
        'retired'
    ))

    conn.commit()
    cur.close()
    conn.close()
```

**Calibrate an item (pretesting → operational):**

```python
def calibrate_item(
    item_id: str,
    new_irt_a: float,
    new_irt_b: float,
    new_irt_c: float,
    se_a: float,
    se_b: float,
    se_c: float,
    n_responses: int,
    fit_passed: bool,
    anchor_items: list
):
    """Update item with calibrated IRT parameters."""

    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    # Get old parameters
    cur.execute("SELECT irt_a, irt_b, irt_c FROM items WHERE id = %s", (item_id,))
    old_a, old_b, old_c = cur.fetchone()

    # Update item
    if fit_passed:
        new_status = 'operational'
    else:
        new_status = 'retired'

    cur.execute("""
        UPDATE items
        SET status = %s,
            updated_at = %s,
            irt_a = %s,
            irt_b = %s,
            irt_c = %s,
            irt_source = 'calibrated',
            calibrated_at = %s,
            n_responses_at_calibration = %s,
            se_irt_a = %s,
            se_irt_b = %s,
            se_irt_c = %s
        WHERE id = %s
    """, (
        new_status,
        datetime.utcnow(),
        new_irt_a,
        new_irt_b,
        new_irt_c,
        datetime.utcnow(),
        n_responses,
        se_a,
        se_b,
        se_c,
        item_id
    ))

    # Log calibration
    cur.execute("""
        INSERT INTO calibration_log (
            log_id, item_id, timestamp,
            old_irt_a, old_irt_b, old_irt_c,
            new_irt_a, new_irt_b, new_irt_c,
            se_irt_a, se_irt_b, se_irt_c,
            n_responses, fit_passed,
            anchor_items_used
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        item_id,
        datetime.utcnow(),
        old_a, old_b, old_c,
        new_irt_a, new_irt_b, new_irt_c,
        se_a, se_b, se_c,
        n_responses,
        fit_passed,
        anchor_items
    ))

    # Log to audit
    cur.execute("""
        INSERT INTO audit_log (log_id, item_id, timestamp, action_type, actor_type, before_status, after_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        item_id,
        datetime.utcnow(),
        'calibrated',
        'system',
        'pretesting',
        new_status
    ))

    conn.commit()
    cur.close()
    conn.close()
```

---

### Domain Coverage Queries

**Check domain distribution:**

```python
def get_domain_coverage(section: str):
    """Get item counts by domain and difficulty."""

    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    query = """
        SELECT
            domain,
            difficulty,
            status,
            COUNT(*) as count
        FROM items
        WHERE section = %s
        GROUP BY domain, difficulty, status
        ORDER BY domain, difficulty, status
    """

    cur.execute(query, (section,))
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results
```

**Find deficit domains (below target count):**

```python
def get_deficit_domains(section: str, target_count: int = 150):
    """Find domains with fewer than target items."""

    conn = psycopg2.connect("postgresql://user:password@localhost/meridian")
    cur = conn.cursor()

    query = """
        SELECT
            domain,
            difficulty,
            COUNT(*) as current_count,
            %s as target_count,
            %s - COUNT(*) as deficit
        FROM items
        WHERE section = %s AND status IN ('draft', 'pretesting', 'operational')
        GROUP BY domain, difficulty
        HAVING COUNT(*) < %s
        ORDER BY deficit DESC
    """

    cur.execute(query, (target_count, target_count, section, target_count))
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results
```

---

## Implementation Details

### Directory Structure

```
src/item_bank/
├── __init__.py
├── database.py            # Database connection management
├── models.py              # SQLAlchemy models (optional)
├── repositories/
│   ├── __init__.py
│   ├── item_repository.py # CRUD operations for items
│   ├── review_repository.py # Review records
│   └── calibration_repository.py # Calibration logs
├── services/
│   ├── __init__.py
│   ├── item_service.py    # High-level item operations
│   └── analytics_service.py # Domain coverage, statistics
└── utils.py

scripts/
├── init_db.py             # Initialize database schema
├── query_items.py         # Query items
├── approve_item.py        # Approve/reject items
└── export_items.py        # Export items for test assembly
```

### Database Connection

**Using connection pooling:**

```python
import psycopg2
from psycopg2 import pool

class Database:
    """Database connection manager."""

    def __init__(self):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host="localhost",
            database="meridian",
            user="meridian_user",
            password="password"
        )

    def get_connection(self):
        """Get a connection from the pool."""
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return a connection to the pool."""
        self.connection_pool.putconn(conn)

    def close_all(self):
        """Close all connections in the pool."""
        self.connection_pool.closeall()

# Global instance
db = Database()
```

---

### Repository Pattern

**Item repository for CRUD operations:**

```python
class ItemRepository:
    """Repository for item CRUD operations."""

    def __init__(self, db):
        self.db = db

    def create(self, item: dict, qa_result: dict) -> str:
        """Create a new item."""
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            # ... (INSERT logic)
            conn.commit()
            return item_id
        finally:
            self.db.return_connection(conn)

    def get_by_id(self, item_id: str):
        """Get item by ID."""
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM items WHERE id = %s", (item_id,))
            return cur.fetchone()
        finally:
            self.db.return_connection(conn)

    def update_status(self, item_id: str, new_status: str):
        """Update item status."""
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE items SET status = %s, updated_at = %s WHERE id = %s",
                (new_status, datetime.utcnow(), item_id)
            )
            conn.commit()
        finally:
            self.db.return_connection(conn)

    def query(self, filters: dict):
        """Query items with filters."""
        # Build dynamic query based on filters
        pass
```

---

## Best Practices

### 1. Always Use Transactions

**Never perform multiple updates without a transaction.**

```python
# ❌ WRONG: No transaction
cur.execute("UPDATE items SET status = 'pretesting' WHERE id = %s", (item_id,))
cur.execute("INSERT INTO review_records ...")  # If this fails, item is inconsistent

# ✅ RIGHT: Use transaction
try:
    cur.execute("UPDATE items SET status = 'pretesting' WHERE id = %s", (item_id,))
    cur.execute("INSERT INTO review_records ...")
    conn.commit()  # Commit only if both succeed
except:
    conn.rollback()  # Rollback if either fails
    raise
```

### 2. Log Every State Change

**The audit log is your source of truth for what happened and when.**

```python
# Log to audit on every status change
cur.execute("""
    INSERT INTO audit_log (log_id, item_id, timestamp, action_type, actor_type, before_status, after_status)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", (...))
```

### 3. Use Indexes for Common Queries

**Add indexes for queries you run frequently.**

```sql
-- Common query: Get draft items for a section
CREATE INDEX idx_items_section_status ON items(section, status);

-- Common query: Get items by domain and difficulty
CREATE INDEX idx_items_domain_difficulty ON items(domain, difficulty);

-- Common query: Full-text search
CREATE INDEX idx_items_content_search ON items USING gin(to_tsvector('english', content_json));
```

### 4. Never Hard-Delete Items

**Items are never deleted, only retired.**

**Why:** Audit trail, calibration history, ability to investigate issues.

```sql
-- ❌ WRONG: Hard delete
DELETE FROM items WHERE id = 'uuid-123';

-- ✅ RIGHT: Soft delete (retire)
UPDATE items SET status = 'retired', retired_at = NOW() WHERE id = 'uuid-123';
```

### 5. Validate Foreign Keys

**Ensure domain values match the domains table.**

```python
def validate_domain(domain: str) -> bool:
    """Check if domain exists in domains table."""
    cur.execute("SELECT 1 FROM domains WHERE name = %s", (domain,))
    return cur.fetchone() is not None
```

---

## Troubleshooting

### Problem: Database Connection Fails

**Symptoms:** `psycopg2.OperationalError: could not connect to server`

**Solutions:**

1. **Check PostgreSQL is running**
   ```bash
   sudo service postgresql status
   sudo service postgresql start
   ```

2. **Check connection string**
   ```python
   # Wrong: "postgresql://user@localhost/db"
   # Right: "postgresql://user:password@localhost:5432/db"
   ```

3. **Check firewall**
   ```bash
   sudo ufw allow 5432/tcp
   ```

### Problem: Trigger Errors on Status Update

**Symptoms:** `ERROR: transition not allowed`

**Causes:**
- Violating state transition rules
- Missing review records for approval

**Solutions:**

1. **Check transition is valid**
   - `draft → pretesting`: Requires review record
   - `pretesting → operational`: Requires calibration

2. **Create missing review records**
   ```python
   # If approving, ensure review record exists first
   cur.execute("INSERT INTO review_records ...")
   cur.execute("UPDATE items SET status = 'pretesting' ...")
   ```

---

## Summary

The Item Bank is the persistent data layer that stores all items and tracks their lifecycle from generation to operational use.

**Key takeaways:**

1. **Four lifecycle states:** draft → pretesting → operational → retired
2. **State transitions are guarded** — enforced by database triggers
3. **Items are never hard-deleted** — only retired (retained for audit)
4. **IRT parameters evolve:** seeded → predicted → calibrated
5. **All changes are logged** — audit_log tracks every state transition
6. **PostgreSQL features:** JSONB for content, full-text search, relational integrity

**Next steps:**
- Read [`docs/00-introduction.md`](./00-introduction.md) for system overview
- Set up PostgreSQL and run `scripts/init_db.py` to initialize schema

---

**Document version:** 1.0
**Last updated:** 2026-03-27
**Author:** IIAS Development Team
