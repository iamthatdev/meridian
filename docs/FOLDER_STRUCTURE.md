The project is structured around a clear **linear pipeline**: configure → generate data → validate → split → train → evaluate → deploy. Every folder plays a role in exactly one stage of that flow.

---

## `configs/`

This is where the pipeline's environment and model behavior are defined. `local.yaml` and `production.yaml` are environment-level configs — they tell the system *where* it's running (your M4 Mac vs. a vast.ai GPU instance), which trainer to use, and which model to load. The distinction matters because on your local machine you use MLX with a tiny proxy model just to validate the pipeline logic, whereas on production you use CUDA with the real Qwen2.5-7B and phi-4 models.

`configs/models/rw.yaml` and `configs/models/math.yaml` are section-level configs. These hold things like LoRA rank, target modules, learning rate, batch size, and the data schema for that section. Keeping them separate is important — phi-4 and Qwen2.5 have different architectures, so their LoRA target module names differ, and Math vs. RW data has a different expected schema and rubric.

---

## `data/`

This folder represents the **stages of data maturity**, not a flat collection of files. Think of it as an assembly line:

`raw/` holds any untouched source material you start from — possibly real SAT questions, reference documents, or seed examples. Nothing here is modified. It's gitignored because it may contain licensed content.

`generated/` is the staging area. When `generator.py` calls the LLM to synthesize training examples, they land here first, unvalidated. It's gitignored because it's cheap to regenerate and can be noisy.

`validated/` holds only the examples that passed auto-QA. The validator has checked schema correctness, rubric scores, and deduplication. This is the first folder in the pipeline where you'd trust the data.

`splits/` is the final output of the data pipeline — clean JSONL files divided into train, val, and test. These are what the trainer actually reads. Keeping splits separate from validated data means you can re-split without re-validating.

---

## `src/data/`

These three files are the engine behind the `data/` folder stages.

`generator.py` handles prompt construction and the actual LLM call. It takes a section (rw or math), builds a prompt that tells the model what kind of example to generate, calls the LLM, and writes raw output to `data/generated/`.

`validator.py` is your quality gate. It runs schema checks (does the example have all required fields?), rubric scoring (is the reasoning sound, is the answer correct?), and deduplication (is this example too similar to one already in the set?). Anything that fails gets dropped.

`pipeline.py` orchestrates the two above in sequence — generate a batch, validate it, then write passing examples into `data/validated/` and finally produce the train/val/test splits in `data/splits/`. When you run `scripts/generate_data.py`, you're calling this pipeline.

---

## `src/models/`

This is where the two fine-tuning targets are defined as Python objects.

`base.py` defines a `ModelConfig` dataclass and a config loader. Every model config — regardless of section — shares common fields like model ID, dtype, and LoRA rank. This is also where the config YAML gets parsed and validated into a typed object that the rest of the code can trust.

`rw.py` and `math.py` each inherit from base and override or extend what's specific to their model. This might include the tokenizer's chat template, section-specific prompt formatting, or any preprocessing logic that differs between Qwen2.5 and phi-4. The trainer doesn't need to know which model it's dealing with — it just calls methods defined on the base interface, and the subclass handles the specifics.

---

## `src/training/`

This is the most important directory for the actual fine-tuning, and it's designed around a hardware abstraction.

`trainer.py` defines an abstract base trainer and a factory function. The factory function reads the environment config and returns either an `MLXTrainer` or a `CUDATrainer`. The rest of the codebase calls `get_trainer(config)` and doesn't need to know which one it got. This is what lets the same `scripts/train.py` command work on your laptop and on a vast.ai A100 without any code changes.

`mlx_trainer.py` is the local implementation. It runs on your M4 Mac using Apple's MLX framework with the small 1.5B proxy model. It won't produce a production-quality adapter, but it lets you verify that the training loop, data loading, and checkpoint saving all work correctly before spending GPU hours on vast.ai.

`cuda_trainer.py` is the real implementation. It uses HuggingFace's `Trainer` class with PEFT for LoRA adapter injection. This is where the actual fine-tuning happens — it loads the base model (phi-4 or Qwen2.5 depending on section), attaches LoRA adapters to the target modules from the model config, runs the training loop on the splits from `data/splits/`, and saves checkpoints to `outputs/checkpoints/rw/` or `outputs/checkpoints/math/`. This is also where the conditional QLoRA logic lives — if the detected VRAM is below 40GB, it loads the base model in 4-bit; otherwise it runs full bf16 LoRA.

---

## `src/evaluation/`

`metrics.py` aggregates scores across a full eval run — accuracy, F1, and rubric score distributions. These are what you look at to decide whether a checkpoint is good enough to deploy.

`auto_qa.py` is called at the *per-example* level during data generation, not just at eval time. It's what `validator.py` calls internally to score individual examples. Having it as a separate module means you can also call it during evaluation to audit model outputs using the same rubric that was used to filter training data — which keeps your quality signal consistent end to end.

---

## `scripts/`

These are the four CLI entry points that a human actually runs. They are thin wrappers — they parse arguments, load the right config, and call into `src/`. The actual logic lives in `src/`, not here.

The intended order of operations is: `generate_data.py` → `train.py` → `evaluate.py` → `deploy.py`. Each script accepts a `--section` flag (rw or math) so you can run the full pipeline independently for each model.

---

## `outputs/`

This folder is entirely gitignored and machine-generated. `checkpoints/` holds the saved LoRA adapter weights (not the full model — just the adapter deltas, which are small). `logs/` holds training loss curves and other metrics emitted during the run. `evals/` holds the output of `evaluate.py` — per-example scores and aggregate metrics that you'd review before deciding to deploy a checkpoint.

---

## How it all connects for a single fine-tune

To fine-tune the Math model from scratch the full sequence would be: set your environment config, run `generate_data.py --section math` which calls `pipeline.py` to produce validated splits, run `train.py --section math` which loads `configs/models/math.yaml`, instantiates phi-4 via `src/models/math.py`, gets a `CUDATrainer`, attaches LoRA adapters, trains on `data/splits/math_train.jsonl`, and saves to `outputs/checkpoints/math/`. You then run `evaluate.py --section math --checkpoint outputs/checkpoints/math/latest` to get metrics, and if they look good, `deploy.py` pushes the adapter to the serving endpoint where `src/api/server.py` loads it for inference.
