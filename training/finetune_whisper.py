#!/usr/bin/env python
"""Fine-tune Whisper on local ASR manifests (JSONL)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import librosa
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], np.ndarray]]]
    ) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_dataset(
    batch: Dict[str, Any],
    processor: WhisperProcessor,
    max_label_length: int,
) -> Dict[str, Any]:
    audio_array, sr = librosa.load(batch["audio"], sr=16000, mono=True)
    audio = {"array": audio_array, "sampling_rate": sr}
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_label_length,
    ).input_ids
    return batch


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(
                {
                    "audio": str(item["audio"]),
                    "text": str(item["text"]),
                }
            )
    if not records:
        raise ValueError(f"No records in manifest: {path}")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper from JSONL manifests.")
    parser.add_argument("--train-manifest", required=True, help="Path to train.jsonl")
    parser.add_argument("--val-manifest", required=True, help="Path to val.jsonl")
    parser.add_argument("--model-name", default="openai/whisper-tiny", help="Base Whisper checkpoint")
    parser.add_argument("--language", default="ru", help="Tokenizer language")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--output-dir", default="training/whisper_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=float, default=5.0, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Per-device eval batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max train steps (-1 to disable)")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 if CUDA is available")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation/save steps")
    parser.add_argument("--logging-steps", type=int, default=25, help="Logging steps")
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=448,
        help="Max decoder label length in tokens.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    raw_datasets = DatasetDict(
        {
            "train": Dataset.from_list(_read_jsonl(args.train_manifest)),
            "validation": Dataset.from_list(_read_jsonl(args.val_manifest)),
        }
    )

    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task=args.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    model.generation_config.suppress_tokens = []

    vectorized = raw_datasets.map(
        lambda batch: prepare_dataset(
            batch, processor, max_label_length=args.max_label_length
        ),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=1,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        fp16=bool(args.fp16),
        eval_strategy="steps",
        save_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized["train"],
        eval_dataset=vectorized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    metrics = trainer.evaluate()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
