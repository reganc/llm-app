"""Unsloth QLoRA fine-tune background jobs."""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import verify_api_key
from config import FINETUNE_DIR

log = logging.getLogger("llm-api.finetune")

router = APIRouter(prefix="/v1/finetune", tags=["finetune"], dependencies=[Depends(verify_api_key)])

_jobs: dict[str, dict] = {}


class FineTuneRequest(BaseModel):
    dataset_path: str
    base_model: str = "mistralai/Mistral-7B-v0.3"
    output_name: str = "mistral-finetuned"
    epochs: int = Field(default=3, ge=1, le=10)
    lora_r: int = Field(default=16, ge=4, le=64)
    max_seq_length: int = Field(default=2048, le=4096)


@router.post("/trigger")
async def trigger_finetune(req: FineTuneRequest):
    if not Path(req.dataset_path).exists():
        raise HTTPException(400, f"Dataset not found: {req.dataset_path}")
    job_id = uuid.uuid4().hex[:10]
    output_dir = Path(FINETUNE_DIR) / job_id
    _jobs[job_id] = {
        "job_id": job_id, "status": "queued",
        "dataset": req.dataset_path, "base_model": req.base_model,
        "output_name": req.output_name, "output_dir": str(output_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None, "log": [], "error": None, "gguf_path": None,
    }
    asyncio.create_task(_run(job_id, req, output_dir))
    return {"job_id": job_id, "status": "queued",
            "poll_url": f"/v1/finetune/status/{job_id}"}


@router.get("/status")
async def list_jobs():
    return {"jobs": list(_jobs.values())}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


def count() -> int:
    return len(_jobs)


_TEMPLATE = """
import sys
sys.path.insert(0, '/app')
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

print("Loading base model: {base_model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model}",
    max_seq_length={max_seq_length},
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r={lora_r},
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha={lora_r},
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)
print("Loading dataset: {dataset_path}")
dataset = load_dataset("json", data_files="{dataset_path}", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length={max_seq_length},
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs={epochs},
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="{output_dir}",
        save_strategy="epoch",
    ),
)
print("Training started...")
trainer.train()
print("Saving GGUF (q4_k_m)...")
model.save_pretrained_gguf(
    "{output_dir}/{output_name}",
    tokenizer,
    quantization_method="q4_k_m",
)
print("Done.")
"""


async def _run(job_id: str, req: FineTuneRequest, output_dir: Path):
    job = _jobs[job_id]
    job["status"] = "running"
    output_dir.mkdir(parents=True, exist_ok=True)
    script = _TEMPLATE.format(
        base_model=req.base_model,
        max_seq_length=req.max_seq_length,
        lora_r=req.lora_r,
        dataset_path=req.dataset_path,
        epochs=req.epochs,
        output_dir=output_dir,
        output_name=req.output_name,
    )
    script_path = output_dir / "train.py"
    script_path.write_text(script)
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            decoded = line.decode().rstrip()
            job["log"].append(decoded)
            if len(job["log"]) > 200:
                job["log"] = job["log"][-200:]
        await proc.wait()
        if proc.returncode == 0:
            job["status"] = "completed"
            job["gguf_path"] = f"{output_dir}/{req.output_name}-unsloth.Q4_K_M.gguf"
        else:
            job["status"] = "failed"
            job["error"] = f"Process exited with code {proc.returncode}"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        log.exception("finetune crashed")
    finally:
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
