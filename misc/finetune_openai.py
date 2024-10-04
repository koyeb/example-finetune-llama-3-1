"""
This script is used to fine-tune OpenAI's models on our custom Apple MLX QA
dataset. We can then use the fine-tuned OpenAI model and compare its performance
with our fine-tuned LLaMa model.
"""

import argparse
import json

import datasets
from datasets import DatasetDict
from openai import OpenAI
from openai.types.fine_tuning.job_create_params import (
    Hyperparameters,
    Integration,
    IntegrationWandb,
)

ASSISTANT_SYSTEM_PROMPT = "You are a helpful AI coding assistant with expert knowledge of Apple's latest machine learning framework: MLX. You can help answer questions about MLX, provide code snippets, and help debug code."

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--lr-multiplier", type=float, default=1.8)
parser.add_argument("--wandb-project", type=str, default="openai-sft-apple-mlx-qa")
args = parser.parse_args()


client = OpenAI()

dataset: DatasetDict = datasets.load_dataset("rojas-diego/apple-mlx-qa")  # type: ignore


def dataset_split_to_fine_tuning_file(split: str) -> tuple[str, bytes]:
    return (
        f"apple-mlx-qa.{split}.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                            {"role": "user", "content": sample["question"]},  # type: ignore
                            {"role": "assistant", "content": sample["answer"]},  # type: ignore
                        ]
                    }
                )
                for sample in dataset[split]
            ]
        ).encode("utf-8"),
    )


training_file = client.files.create(
    file=dataset_split_to_fine_tuning_file("train"),
    purpose="fine-tune",
)

validation_file = client.files.create(
    file=dataset_split_to_fine_tuning_file("test"),
    purpose="fine-tune",
)

client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters=Hyperparameters(
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate_multiplier=args.lr_multiplier,
    ),
    integrations=[
        Integration(type="wandb", wandb=IntegrationWandb(project="apple-mlx-qa"))
    ],
)
