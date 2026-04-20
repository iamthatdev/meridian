# AWS Migration Architecture Diagrams

This directory contains the architecture diagrams for the AWS training infrastructure migration.

## Diagrams

### 1. Architecture Overview (`architecture-overview.dot`)
**High-level system architecture showing the flow from GitHub Actions to AWS Cloud to EC2 instances and S3 storage.**

**Key Components:**
- GitHub Actions (orchestration layer)
- AWS Cloud with CDK stack
- EC2 Spot Instances (training compute)
- S3 Storage (data layer)

**View:**
```bash
dot -Tpng architecture-overview.dot -o architecture-overview.png
```

---

### 2. Spot Interruption Flow (`spot-interruption-flow.dot`)
**Detailed flow of spot instance interruption handling and automatic recovery.**

**Key Stages:**
1. AWS sends 2-minute interruption warning
2. Instance Metadata Service detects warning
3. Background daemon catches SIGTERM signal
4. Training process saves checkpoint to S3
5. CloudWatch Events detects termination
6. Auto Scaling Group launches replacement instance
7. New instance resumes training from checkpoint

**View:**
```bash
dot -Tpng spot-interruption-flow.dot -o spot-interruption-flow.png
```

---

### 3. Cost Monitoring Stages (`cost-monitoring-stages.dot`)
**Three-stage cost monitoring system: pre-training estimation, real-time tracking, post-training breakdown.**

**Stages:**
- **Stage 1:** Pre-training cost estimation (85% confidence)
- **Stage 2:** Real-time cost tracking (updates every 60s)
- **Stage 3:** Post-training cost breakdown (actual vs estimate vs on-demand)

**View:**
```bash
dot -Tpng cost-monitoring-stages.dot -o cost-monitoring-stages.png
```

---

### 4. CDK Stack Structure (`cdk-stack-structure.dot`)
**CDK infrastructure components and their relationships.**

**Key Constructs:**
- VpcConstruct (VPC, subnets, NAT, S3 endpoints)
- S3Construct (data, checkpoint, model buckets with lifecycle policies)
- IAMConstruct (EC2 role, S3/CloudWatch/Cost Explorer access)
- EC2Construct (launch template, ASG, spot fleet, user data)
- MonitoringConstruct (CloudWatch dashboard, SNS, metrics, alarms)
- SecurityConstruct (security groups, KMS, IAM policies)

**View:**
```bash
dot -Tpng cdk-stack-structure.dot -o cdk-stack-structure.png
```

---

### 5. GitHub Actions Workflow (`github-actions-workflow.dot`)
**Complete training workflow from trigger to cleanup.**

**Workflow Steps:**
1. Cost estimation (pre-flight check)
2. Infrastructure validation
3. Data validation
4. Create job manifest (upload to S3)
5. Start EC2 instance (ASG 0→1)
6. Monitor training (cost_tracker.py)
7. Wait for completion (poll S3)
8. Generate final report
9. Cleanup (ASG back to 0)

**View:**
```bash
dot -Tpng github-actions-workflow.dot -o github-actions-workflow.png
```

---

### 6. S3 Data Flow (`s3-data-flow.dot`)
**Data flow from local development through S3 to EC2 training instances.**

**Key Buckets:**
- `meridian-training-data/` - Training/validation/test datasets
- `meridian-checkpoints/` - Checkpoint versions every 500 steps
- `meridian-models-prod/` - Final trained models
- `meridian-training-jobs/` - Job manifests, logs, reports

**View:**
```bash
dot -Tpng s3-data-flow.dot -o s3-data-flow.png
```

---

### 7. Migration Timeline (`migration-timeline.dot`)
**16-day migration plan across 3 weeks.**

**Phases:**
- **Phase 1 (Days 1-3):** Foundation - AWS setup, S3 buckets, data migration, GPU AMI
- **Phase 2 (Days 4-6):** CDK Infrastructure - Deploy stack, test infrastructure
- **Phase 3 (Days 7-9):** Script Migration - Cost estimator, spot handler, cost tracker
- **Phase 4 (Days 10-12):** GitHub Actions - Configure secrets, create workflows, test E2E
- **Phase 5 (Days 13-15):** Parallel Testing - RunPod vs AWS comparison
- **Phase 6 (Day 16):** Cutover - Final verification, decommission RunPod

**View:**
```bash
dot -Tpng migration-timeline.dot -o migration-timeline.png
```

---

## How to View Diagrams

### Option 1: Generate PNG Images
Requires Graphviz to be installed:

```bash
# macOS
brew install graphviz

# Generate all diagrams
for file in *.dot; do
  dot -Tpng "$file" -o "${file%.dot}.png"
done
```

### Option 2: Online Viewer
1. Go to https://dreampuf.github.io/GraphvizOnline/
2. Copy the contents of any `.dot` file
3. Paste into the online viewer
4. Adjust layout if needed (dot, neato, circo, etc.)
5. Export as PNG/SVG

### Option 3: VS Code Extension
1. Install "Graphviz Interactive Preview" extension
2. Open any `.dot` file
3. Preview panel shows the diagram
4. Updates automatically as you edit

### Option 4: Command Line (Quick View)
```bash
# View as ASCII art (requires graphviz)
dot -Tpng architecture-overview.dot | display -

# Or generate SVG for web viewing
dot -Tsvg architecture-overview.dot -o architecture-overview.svg
```

---

## Diagram Legend

| Color | Meaning |
|-------|---------|
| 🟢 Green (#2EA44F) | GitHub Actions, Local, Success |
| 🟡 Yellow (#FF9900) | AWS, EC2, Warning/Action |
| 🔵 Blue (#232F3E) | AWS Infrastructure, Compute |
| 🟠 Light Green (#E8F5E9) | Background Processes |
| 🔴 Light Blue (#E3F2FD) | AWS Services |
| 🟠 Light Yellow (#FFF9C4) | Outputs, Reports, Notes |

---

## Updating Diagrams

1. Edit the `.dot` file with your changes
2. Regenerate the PNG/SVG: `dot -Tpng file.dot -o file.png`
3. Commit both the `.dot` source and generated image

**Why DOT format?**
- Human-readable text format
- Version control friendly
- Easy to modify
- Can generate multiple output formats (PNG, SVG, PDF)

---

## Related Documentation

- [Main Design Document](../2024-04-20-aws-training-infrastructure-design.md)
- [CLAUDE.md](../../../CLAUDE.md)
- [Migration Plan](../MIGRATION_PLAN.md) (to be created)

---

**Last Updated:** 2024-04-20
**Tool:** Graphviz DOT language
