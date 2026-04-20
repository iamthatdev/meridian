# AWS Training Infrastructure Design

**Project:** Meridian AWS Migration
**Date:** 2024-04-20
**Status:** Design Approved
**Author:** Claude (with user collaboration)

---

## Executive Summary

This document outlines the design for migrating Meridian's training infrastructure from RunPod to AWS using EC2 Spot instances with S3 storage, CDK for infrastructure-as-code, and GitHub Actions for orchestration.

**Key Benefits:**
- **Cost Savings:** 70-90% discount via Spot instances (vs 30-50% on RunPod)
- **Reliability:** Auto-recovery from spot interruption with checkpoint/resume
- **Visibility:** Three-stage cost monitoring (estimate, real-time, final breakdown)
- **GitOps:** Full CI/CD integration via GitHub Actions
- **Scalability:** Easy to add capacity or new instance types

**Migration Timeline:** 16 days (~2.5 weeks)
**Risk Level:** Low (parallel testing, phased rollout)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Spot Interruption Handling](#spot-interruption-handling)
3. [Cost Monitoring System](#cost-monitoring-system)
4. [CDK Infrastructure Components](#cdk-infrastructure-components)
5. [GitHub Actions Workflow](#github-actions-workflow)
6. [Data Flow & S3 Integration](#data-flow--s3-integration)
7. [Migration Path](#migration-path)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions                              │
│                  (Orchestration Layer)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Validate data in S3                                   │  │
│  │ 2. Trigger EC2 Spot fleet via AWS SDK                     │  │
│  │ 3. Monitor training via CloudWatch Logs                   │  │
│  │ 4. Sync checkpoints from S3 when complete                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    CDK Stack                              │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │  │
│  │  │ VPC +       │  │ EC2 Spot     │  │ IAM Roles &    │  │  │
│  │  │ Subnets     │  │ Fleet        │  │ Security Grps  │  │  │
│  │  └─────────────┘  └──────────────┘  └────────────────┘  │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │  │
│  │  │ S3 Buckets  │  │ CloudWatch   │  │ Auto Scaling   │  │  │
│  │  │ (Data/Mdl)  │  │ (Logs/Metrics)│  │ Groups         │  │  │
│  │  └─────────────┘  └──────────────┘  └────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EC2 Spot Instances                            │
│              (Training Compute Layer)                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Boot from AMI with CUDA + dependencies                  │  │
│  │ • Pull training script from S3                            │  │
│  │ • Stream data from S3 during training                     │  │
│  │ • Sync checkpoints to S3 every N minutes                  │  │
│  │ • Handle spot interruption gracefully                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         S3 Storage                               │
│              (Data Layer - Single Source of Truth)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │ meridian-    │  │ meridian-    │  │ meridian-        │      │
│  │ training-    │  │ checkpoints/ │  │ models-          │      │
│  │ data/        │  │              │  │ prod/            │      │
│  │              │  │              │  │                  │      │
│  │ • train/     │  │ • math/      │  │ • math-sft-      │      │
│  │ • val/       │  │ • rw/        │  │   v1.0/          │      │
│  │ • test/      │  │              │  │ • rw-sft-        │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **S3 as single source of truth** - All data flows through S3, no local storage dependencies
2. **Spot fleet with diversification** - Use multiple instance types to increase availability
3. **Stateless training** - Everything needed is pulled from S3, nothing stored locally
4. **Checkpoints as recovery mechanism** - Auto-save every N minutes, resume on interruption
5. **GH Actions as orchestrator, not runner** - GH triggers jobs, doesn't run them

### Cost Optimization Priorities

1. Spot instances (70-90% discount vs on-demand)
2. Right-sized instances (no over-provisioning)
3. Auto-termination when training completes
4. S3 lifecycle policies for old checkpoints
5. No SageMaker surcharges

---

## Spot Interruption Handling

### Interruption Flow

```
┌─────────────────────────────────────────────────────────────────┐
│           Spot Instance Interruption Flow                       │
└─────────────────────────────────────────────────────────────────┘

1. AWS sends 2-minute interruption warning
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ EC2 Instance Metadata Service detects warning              │
│ http://169.254.169.254/latest/meta-data/spot/instance-action│
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Background daemon catches signal                           │
│ • SIGTERM from instance metadata                          │
│ • Triggers graceful shutdown sequence                     │
└───────────────────────────────────────────────────────────┘
        │
        ├──► Training process receives SIGTERM
        │        │
        │        ▼
        │  ┌───────────────────────────────────────────┐
        │  │ 1. Finish current step (max 30 sec)      │
        │  │ 2. Save checkpoint to S3                 │
        │  │ 3. Upload training logs                  │
        │  │ 4. Write recovery state to S3            │
        │  │    (last epoch, step, metrics)           │
        │  └───────────────────────────────────────────┘
        │
        ├──► CloudWatch Events detects instance termination
        │        │
        │        ▼
        │  ┌───────────────────────────────────────────┐
        │  │ Auto Scaling Group automatically         │
        │  │ launches replacement spot instance       │
        │  └───────────────────────────────────────────┘
        │
        └──► New instance boots
                 │
                 ▼
        ┌───────────────────────────────────────────┐
        │ User data script on boot:                 │
        │ 1. Check S3 for incomplete training       │
        │ 2. Download latest checkpoint             │
        │ 3. Resume from exact step/epoch           │
        │ 4. Continue training                      │
        └───────────────────────────────────────────┘
```

### Implementation Components

**1. Interruption Handler (`spot_handler.py`):**
```python
# Runs as background daemon on every instance
# Polls instance metadata for interruption notice
# When detected:
#   - Send SIGTERM to training process
#   - Wait up to 90s for graceful shutdown
#   - Force kill if still running (2min safety buffer)
#   - Sync final checkpoint to S3
```

**2. Training Script Modifications:**
```python
# In train_huggingface.py
import signal

def graceful_shutdown(signum, frame):
    logger.info("Spot interruption detected, saving checkpoint...")
    trainer.save_checkpoint_to_s3()  # Custom method
    upload_logs_to_s3()
    save_recovery_state_to_s3()  # epoch, step, metrics
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
```

**3. Auto-Resume Logic:**
```python
# In user data boot script
s3_recovery_state = check_s3_for_incomplete_training()
if s3_recovery_state:
    resume_from = s3_recovery_state["checkpoint_path"]
    logger.info(f"Resuming from checkpoint: {resume_from}")
    train(resume_from=resume_from)
```

**4. S3 Recovery State Structure:**
```
s3://meridian-checkpoints/
  ├── math/
  │   ├── 2024-04-20_training_run/
  │   │   ├── recovery_state.json  {"status": "incomplete",
  │   │   │                         "last_checkpoint": "...",
  │   │   │                         "last_epoch": 2,
  │   │   │                         "last_step": 1234}
  │   │   ├── checkpoint-500/
  │   │   ├── checkpoint-1000/
  │   │   └── checkpoint-1500/  ← Resume from here
```

### Auto Scaling Group Configuration

- **Desired capacity:** 1 instance per training job
- **Min/Max:** 0 / 3 (allow brief overlap during replacement)
- **Replacement strategy:** Launch before terminate (if capacity allows)
- **Multiple instance types:** Mix of p3.2xlarge, p3.8xlarge, g4dn.2xlarge

### Cost Impact

- Interruption = max 2 minutes of wasted compute
- Checkpoint sync = ~30 seconds to S3
- Replacement instance launch = ~2-3 minutes
- **Total downtime:** ~5 minutes, fully automated

### Reliability Features

- Checkpoint every 500 steps (configurable)
- Final checkpoint saved on interruption
- CloudWatch alarm if training doesn't resume in 10 minutes
- S3 versioning prevents checkpoint corruption

---

## Cost Monitoring System

### Three-Stage Cost Visibility

#### Stage 1: Pre-Training Cost Estimation

```
User runs: python scripts/cost_estimator.py --section math --epochs 3
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Cost Estimator Script                                     │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Inputs:                                              │   │
│ │ • Training config (epochs, batch size, steps)       │   │
│ │ • Model size (7B, 14B)                              │   │
│ │ • Instance type preferences                         │   │
│ │ • Historical spot price data                        │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Queries:                                             │   │
│ │ 1. AWS Spot Price API (last 30 days)                │   │
│ │ 2. Current spot capacity availability               │   │
│ │ 3. Your historical training runs from S3            │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Calculates:                                          │   │
│ │ • Estimated training time (based on history)        │   │
│ │ • Spot price range (p10, p50, p90)                  │   │
│ │ • Instance hours needed                             │   │
│ │ • Storage costs (S3 checkpoints, logs)              │   │
│ │ • Data transfer costs                               │   │
│ └─────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Output: Cost Estimate Report                               │
│ ─────────────────────────────────────────────────────────  │
│ Training Job: Math Model - 3 Epochs                        │
│                                                            │
│ Time Estimate: 4-6 hours (based on historical data)        │
│                                                            │
│ Instance Options (by cost):                                │
│ ┌────────────────┬──────────┬──────────┬──────────────┐   │
│ │ Instance Type  │ Spot $/hr│ Est Cost │ Availability │   │
│ ├────────────────┼──────────┼──────────┼──────────────┤   │
│ │ p3.2xlarge     │ $0.92    │ $4.60    │ High (90%)   │   │
│ │ g4dn.2xlarge   │ $0.75    │ $3.75    │ Medium (60%) │   │
│ │ p3.8xlarge     │ $3.06    │ $15.30   │ Low (30%)    │   │
│ └────────────────┴──────────┴──────────┴──────────────┘   │
│                                                            │
│ Recommended: p3.2xlarge (best availability/cost balance)   │
│                                                            │
│ Cost Breakdown:                                            │
│ • Compute (spot):    $4.60                                 │
│ • Storage (S3):      $0.15                                 │
│ • Data transfer:     $0.05                                 │
│ ─────────────────────────────────────                       │
│ Total Estimated:     $4.80                                 │
│                                                            │
│ Confidence: 85% (based on 5 similar historical runs)       │
└───────────────────────────────────────────────────────────┘
```

#### Stage 2: Real-Time Cost Tracking

```
GitHub Actions Workflow (running)
        │
        ├──► Cost monitoring sidecar starts
        │        │
        │        ▼
        │  ┌───────────────────────────────────────────────────┐
        │  │ CloudWatch Cost Metric Custom Widget              │
        │  │ ─────────────────────────────────────────────────  │
        │  │ Current Job: math-training-2024-04-20-14:30       │
        │  │                                                     │
        │  │ ┌─────────────────────────────────────────────┐   │
        │  │ │ RUNNING: 2h 15m / Est. 5h 30m               │   │
        │  │ │                                               │   │
        │  │ │ Current Cost:        $1.84                   │   │
        │  │ │ Projected Cost:      $4.92                   │   │
        │  │ │                                                │   │
        │  │ │ Spot price:          $0.92/hr                │   │
        │  │ │ Instance:            p3.2xlarge               │   │
        │  │ │ Interruptions:       0                       │   │
        │  │ │                                                │   │
        │  │ │ ████████████░░░░░░░░░░░░░░░░ 40% complete    │   │
        │  │ └─────────────────────────────────────────────┘   │
        │  │                                                     │
        │  │ Last checkpoint: 2m ago (step 1250/2500)           │
        │  │ Est. completion: 3h 15m                            │
        │  │                                                     │
        │  │ [View detailed metrics] [Kill job]                 │
        │  └───────────────────────────────────────────────────┘
        │
        └──► CLI Query (user runs in parallel)
                 $ python scripts/cost_tracker.py --job-id math-training-2024-04-20-14:30

                 Job: math-training-2024-04-20-14:30
                 Status: RUNNING

                 Real-Time Cost Breakdown:
                 ┌──────────────────────────────────────────┐
                 │ Compute:  $1.84 (2h @ $0.92/hr)        │
                 │ Storage:  $0.08 (S3 logs + checkpoints) │
                 │ Transfer: $0.02 (data in/out)          │
                 │ ─────────────────────────────────────── │
                 │ Current:   $1.94                        │
                 │ Projected: $4.92                        │
                 └──────────────────────────────────────────┘

                 Training Progress:
                 • Epoch: 2/3 (66%)
                 • Step: 1250/2500
                 • Loss: 0.234
                 • Est. remaining: 3h 15m

                 Instance Health:
                 • Spot interruptions: 0
                 • Checkpoint age: 2m
                 • GPU utilization: 94%
```

#### Stage 3: Post-Training Cost Breakdown

```
Job completes → CloudWatch triggers SNS notification
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Final Cost Report (saved to S3 + emailed)                 │
│ ─────────────────────────────────────────────────────────  │
│ Training Job: Math Model - 3 Epochs - COMPLETED           │
│ Completion Time: 2024-04-20 19:45 UTC                     │
│                                                            │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ FINAL COST BREAKDOWN                                 │   │
│ │ ─────────────────────────────────────────────────── │   │
│ │                                                      │   │
│ │ Compute:                                             │   │
│ │ • p3.2xlarge (spot):    5h 12m  @ $0.92/hr         │   │
│ │ • Instance hours:       5.2 hours                   │   │
│ │ • Spot savings:         $13.52 (vs on-demand $19.24)│   │
│ │ ─────────────────────────────────────────────────   │   │
│ │ Compute subtotal:        $4.78                       │   │
│ │                                                      │   │
│ │ Storage:                                             │   │
│ │ • S3 checkpoints:       2.3 GB × 5 versions          │   │
│ │ • S3 logs:              156 MB                       │   │
│ │ ─────────────────────────────────────────────────   │   │
│ │ Storage subtotal:        $0.12                       │   │
│ │                                                      │   │
│ │ Data Transfer:                                      │   │
│ │ • Data in:               450 MB                      │   │
│ │ • Checkpoints out:       2.3 GB                      │   │
│ │ ─────────────────────────────────────────────────   │   │
│ │ Transfer subtotal:        $0.07                      │   │
│ │                                                      │   │
│ │ ─────────────────────────────────────────────────   │   │
│ │ TOTAL ACTUAL COST:        $4.97                      │   │
│ │                                                      │   │
│ │ vs Original Estimate:      $4.80 (±3.5%)            │   │
│ │ vs On-Demand Cost:        $19.42 (74% savings)      │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                            │
│ Performance Metrics:                                        │
│ • Final loss:           0.189                              │
│ • Training accuracy:    92.4%                              │
│ • Validation accuracy:  89.7%                              │
│                                                            │
│ Cost Optimization Insights:                                 │
│ ✓ No spot interruptions (ideal run)                        │
│ ✓ Right-sized instance (94% GPU utilization)              │
│ ⚠ Checkpoint frequency could be reduced (save ~$0.03)     │
└───────────────────────────────────────────────────────────┘
```

### Implementation Components

**1. Cost Estimator Script (`scripts/cost_estimator.py`):**
```python
# Uses AWS Price List Query API
# Queries historical spot prices from your account
# Calculates based on training config
# Outputs: JSON report + Markdown summary
```

**2. Cost Tracker CLI (`scripts/cost_tracker.py`):**
```python
# Queries CloudWatch custom metrics in real-time
# Shows current job cost + projection
# Uses AWS Cost Explorer API for actuals
```

**3. CloudWatch Custom Metrics:**
```python
# EC2 instance publishes metrics every 60s:
# - TrainingJobCost (cumulative)
# - TrainingJobProgress (%)
# - SpotInterruptionCount
# - GPUUtilization
```

**4. Cost Anomaly Detection:**
```python
# CloudWatch Alarm: If cost exceeds 120% of estimate
# → SNS notification + stops job
# Prevents runaway spending
```

---

## CDK Infrastructure Components

### Stack Structure

```
aws-training-infra/
├── lib/
│   ├── training-stack.ts           # Main stack
│   ├── constructs/
│   │   ├── vpc-construct.ts        # VPC + networking
│   │   ├── s3-construct.ts         # S3 buckets + lifecycle
│   │   ├── ec2-construct.ts        # EC2 spot fleet + ASG
│   │   ├── iam-construct.ts        # IAM roles + policies
│   │   ├── monitoring-construct.ts # CloudWatch + alarms
│   │   └── security-construct.ts   # Security groups + KMS
│   └── lambda/
│       ├── cost-estimator.ts       # Lambda for cost estimation
│       └── spot-handler.ts         # Lambda for spot interruption handling
├── bin/
│   └── training-stack.ts           # Stack entry point
├── test/
│   └── training-stack.test.ts
├── cdk.json
└── package.json
```

### Key Constructs

#### VpcConstruct
- VPC with private subnets only (no public subnets needed)
- VPC Endpoints for S3 (no NAT gateway cost for S3 traffic)
- 2 AZs for high availability

#### S3Construct
- **Training data bucket:** Versioned, 90-day retention
- **Checkpoint bucket:** Versioned, transitions to IA → Glacier
- **Models bucket:** Versioned, 1-year noncurrent version retention
- **Jobs bucket:** Job manifests, logs, reports

#### EC2Construct
- Launch Template with custom GPU AMI
- Auto Scaling Group for Spot Fleet
- Desired capacity: 0 (scale to 1 when training)
- Max capacity: 3 (allow replacement instances)
- Spot price: $0.92/hr (p3.2xlarge)
- User data script for boot-time configuration

#### IAMConstruct
- EC2 instance role with:
  - S3 read/write access (training data, checkpoints, models)
  - CloudWatch Logs access
  - Cost Explorer access
  - SSM managed instance core (for SSH access)

#### MonitoringConstruct
- CloudWatch Dashboard for cost monitoring
- SNS Topic for alerts (cost anomalies, failures)
- Custom metrics: JobCost, SpotPrice, GPUUtilization
- Alarms: Cost anomaly, training failure, no heartbeat

### Stack Outputs

```
TrainingStack.ASGName = meridian-training-asg-12345
TrainingStack.CheckpointBucket = meridian-checkpoints-abc123
TrainingStack.CostDashboardUrl = https://console.aws.amazon.com/cloudwatch/...
```

---

## GitHub Actions Workflow

### Workflow Files

```
.github/
└── workflows/
    ├── train-math.yml
    ├── train-rw.yml
    ├── cost-monitoring.yml
    └── deploy-infra.yml
```

### Training Workflow Steps

1. **Checkout code** - Clone repository
2. **Configure AWS credentials** - Set up AWS CLI
3. **Set up Python** - Install dependencies
4. **Estimate training cost** - Pre-flight cost check
5. **Validate infrastructure** - Verify CDK stack deployed
6. **Validate training data** - Check S3 for valid data
7. **Create training job manifest** - Upload to S3
8. **Start EC2 Spot Instance** - Update ASG capacity
9. **Monitor training progress** - Real-time cost tracking
10. **Wait for completion** - Poll job status from S3
11. **Generate final report** - Cost + performance summary
12. **Cleanup** - Return ASG to capacity 0

### Workflow Features

- **Manual or automatic triggers** - On-demand or on merge
- **Cost estimation before training** - No surprises
- **Real-time monitoring** - Live cost tracking
- **Hard cost limits** - Auto-stop if budget exceeded
- **Comprehensive logging** - Full audit trail
- **Post-training reports** - Automatic summary generation

### Example Workflow Call

```yaml
name: Train Math Model

on:
  workflow_dispatch:
    inputs:
      epochs:
        description: 'Number of training epochs'
        required: true
        default: '3'
        type: choice
        options:
          - '1'
          - '3'
          - '5'
      instance_type:
        description: 'EC2 instance type'
        required: false
        default: 'p3.2xlarge'
        type: choice
        options:
          - 'p3.2xlarge'
          - 'p3.8xlarge'
          - 'g4dn.2xlarge'
      cost_limit:
        description: 'Maximum cost limit (USD)'
        required: false
        default: '10.00'
```

---

## Data Flow & S3 Integration

### S3 Bucket Structure

```
s3://meridian-training-data/
├── math/
│   ├── train.jsonl          (versioned)
│   ├── val.jsonl
│   └── test.jsonl
└── reading_writing/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl

s3://meridian-checkpoints/
├── math/
│   ├── 2024-04-20-14:30/    (job ID)
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   ├── checkpoint-1500/
│   │   └── final_model/
│   └── 2024-04-21-09:15/    (next job)
└── reading_writing/
    └── ...

s3://meridian-models-prod/
├── math-sft-v1.0/
│   ├── model/
│   ├── eval/
│   └── metadata.json
└── reading_writing-sft-v1.0/

s3://meridian-training-jobs/
├── math-training-2024-04-20-14:30/
│   ├── manifest.json        (job state machine)
│   ├── logs/
│   └── cost_report.json
└── ...

s3://meridian-training-reports/
├── math-training-2024-04-20-14:30/
│   ├── cost_report.json
│   ├── performance_metrics.json
│   └── optimization_insights.txt
└── ...

s3://meridian-training-scripts/  (static, cached locally)
├── train_huggingface.py
├── cost_tracker.py
└── spot_handler.py
```

### Data Streaming Implementation

```python
# In train_huggingface.py

import s3fs
from datasets import load_dataset

def load_data_from_s3(s3_path: str):
    """
    Stream data directly from S3 without local download.
    """
    # Use S3 filesystem for streaming
    fs = s3fs.S3FileSystem(
        key=os.getenv('AWS_ACCESS_KEY_ID'),
        secret=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

    # Load dataset with streaming mode
    dataset = load_dataset(
        "json",
        data_files=s3_path,
        split="train",
        streaming=True  # Don't download entire file
    )

    return dataset
```

### S3 Lifecycle Policies

**Training Data:**
- Standard → IA after 7 days
- IA → Glacier after 30 days
- Expire after 90 days

**Checkpoints:**
- Standard → IA after 7 days
- IA → Glacier after 30 days
- Keep forever (manual cleanup)

**Models:**
- Keep forever (production assets)
- Noncurrent versions expire after 365 days

---

## Migration Path

### Migration Phases

#### Phase 1: Foundation (Days 1-3)
- AWS account setup
- S3 bucket creation
- Data migration to S3
- Custom GPU AMI creation

#### Phase 2: CDK Infrastructure (Days 4-6)
- Initialize CDK project
- Deploy VPC, S3, IAM, EC2, monitoring
- Test infrastructure with manual instance launch
- Verify spot interruption handling

#### Phase 3: Script Migration (Days 7-9)
- Create cost estimator script
- Add spot interruption handling
- Create cost tracker script
- Test scripts locally

#### Phase 4: GitHub Actions Setup (Days 10-12)
- Configure GitHub secrets
- Create workflow files
- Test manual workflow trigger
- Verify end-to-end execution

#### Phase 5: Parallel Testing (Days 13-15)
- Run identical jobs on RunPod and AWS
- Compare results (cost, performance, reliability)
- Validate model outputs are equivalent
- Document any differences

#### Phase 6: Cutover (Day 16)
- Final verification on AWS
- Update documentation
- Archive RunPod artifacts
- Cancel RunPod instances
- Celebrate cost savings! 🎉

### Timeline Summary

```
Week 1: Foundation + S3 setup (Days 1-3) | CDK infrastructure (Days 4-6)
Week 2: Script migration (Days 7-9) | GitHub Actions setup (Days 10-12)
Week 3: Parallel testing (Days 13-15) | Cutover (Day 16)

Total: 16 days (~2.5 weeks)
Risk: Low (parallel testing ensures safety)
Cost savings: ~23-30% vs RunPod
```

### Rollback Plan

**Immediate rollback (Hours 0-24):**
- Stop GitHub Actions workflows
- Resume using RunPod via MLT automation
- Investigate AWS issues

**Partial rollback (Days 1-7):**
- Keep AWS infrastructure running
- Use RunPod for production jobs
- Use AWS for experimental jobs
- Fix issues incrementally

**Full rollback (Days 7+):**
- Decommission AWS infrastructure
- Delete S3 buckets (after backup)
- Return to RunPod exclusively
- Re-evaluate migration strategy

---

## Next Steps

1. **Review this design document** - Confirm all requirements are met
2. **Implement CDK stack** - Deploy infrastructure to AWS
3. **Create supporting scripts** - Cost estimator, tracker, spot handler
4. **Set up GitHub Actions** - Configure workflows and secrets
5. **Begin migration** - Follow migration phases 1-6
6. **Monitor and optimize** - Continuously improve cost and performance

---

## Appendix

### Instance Type Options

| Instance Type | GPU | Spot Price | Memory | Use Case |
|--------------|-----|------------|--------|----------|
| p3.2xlarge | 1×V100 | $0.92/hr | 61 GB | Default recommendation |
| p3.8xlarge | 4×V100 | $3.06/hr | 244 GB | Large models, faster training |
| g4dn.2xlarge | 1×T4 | $0.75/hr | 16 GB | Budget option, slower |

### Cost Comparison

| Platform | Hourly Cost | 3-Epoch Training | Savings |
|----------|-------------|------------------|---------|
| RunPod Spot | $1.20/hr | $6.50 | Baseline |
| AWS Spot (p3.2xlarge) | $0.92/hr | $4.97 | **23%** |
| AWS On-Demand | $3.06/hr | $19.42 | -199% |

### Key Metrics to Monitor

- **Cost:** Actual vs estimate
- **Interruptions:** Count and duration
- **GPU Utilization:** Target >90%
- **Checkpoint Frequency:** Every 500 steps
- **Training Time:** Compare to baseline
- **Model Quality:** Accuracy, F1, loss

---

**Document Version:** 1.0
**Last Updated:** 2024-04-20
**Status:** Ready for Implementation
