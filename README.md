# IIAS - Intelligent Item Authoring System

A machine learning platform that generates, validates, and manages SAT-style test questions.

## Documentation

See `docs/` for complete documentation:
- `docs/00-introduction.md` - System overview
- `docs/QUICKSTART.md` - Quick start guide
- `docs/01-generation-service.md` - Generation Service
- `docs/02-auto-qa-service.md` - Auto-QA Service
- `docs/03-item-bank.md` - Item Bank

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
```

## Development

Run tests:
```bash
pytest tests/ -v
```

See `docs/QUICKSTART.md` for detailed workflow.