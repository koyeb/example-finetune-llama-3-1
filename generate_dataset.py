"""
Generates question-answer pairs from the Apple MLX documentation.

1. Collects chunks of documentation from MLX's API reference.
2. Ask model to generate N candidate questions per chunk.
3. Ask model to answer each question based on the documentation.
4. Output a JSONL of question-answer pairs. Each line is a JSON object with the following keys:
    - question: string
    - answer: string
    - chunk: string
"""

import argparse
import glob
import json
import os

import tqdm
from datasets import Dataset
from openai import OpenAI

# Number of questions to generate per documentation snippet.
N = 3

# The paths inside the documentation build directory where the API documentation
# snippets are located.
API_REFERENCE_PATHS = [
    "python/**/_autosummary/mlx.*.txt",
    "python/**/_autosummary_functions/mlx.*.txt",
]

# The system prompt used to generate questions from a reference documentation
# snippet.
QUESTION_GENERATION_SYSTEM_PROMPT = """You are a helpful AI assistant. Your task is to help a user understand how to use functions and classes from Apple's Deep Learning framework, MLX. Carefully examine the function documentation snippet and generate {} questions a medium to experienced MLX user could ask. Questions must be answerable from the information in the snippet. Do not assume anything about MLX's API that is not discussed in the snippet. If the snippet is too short or contains too little information, output an empty JSON array.""".format(
    N
)

# The system prompt used to generate answers to a question about a documentation
# snippet.
QUESTION_ANSWERING_SYSTEM_PROMPT = """You are a helpful AI assistant. Your task is to help a user understand how to use functions and classes from Apple's Deep Learning framework, MLX. Carefully examine the function documentation and generate an explanatory response based on the user's question which showcases usage and examples. Do not assume anything about MLX's API that is not discussed in the reference documentation snippet."""


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default="qa.jsonl")
parser.add_argument("-i", "--input", type=str, default="mlx/docs/build/text")
parser.add_argument("-m", "--model", type=str, default="gpt-4o")
parser.add_argument("-r", "--repo", type=str, required=True)
args = parser.parse_args()

# Generate question-answer pairs. Skip if the output file already exists.
if not os.path.exists(args.output):
    api_reference_chunks = []
    for wcpath in API_REFERENCE_PATHS:
        for path in glob.glob(os.path.join(args.input, wcpath), recursive=True):
            with open(path) as f:
                api_reference_chunks.append(f.read())
    print(f"Found {len(api_reference_chunks)} chunks of documentation")

    client = OpenAI()

    with open(args.output, "w") as f:
        progress = tqdm.tqdm(api_reference_chunks)
        prompt_tokens_used = 0
        completion_tokens_used = 0
        for chunk in progress:
            completion = client.chat.completions.create(
                model=args.model,
                temperature=0.3,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": QUESTION_ANSWERING_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "questions",
                        "schema": {
                            "type": "object",
                            "required": ["questions"],
                            "properties": {
                                "questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
            questions = json.loads(completion.choices[0].message.content)["questions"]
            prompt_tokens_used += completion.usage.prompt_tokens
            completion_tokens_used += completion.usage.completion_tokens

            for question in questions[:N]:
                completion = client.chat.completions.create(
                    model=args.model,
                    temperature=0.3,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "system",
                            "content": QUESTION_ANSWERING_SYSTEM_PROMPT,
                        },
                        {"role": "assistant", "content": chunk},
                        {"role": "user", "content": question},
                    ],
                )
                answer = completion.choices[0].message.content
                prompt_tokens_used += completion.usage.prompt_tokens
                completion_tokens_used += completion.usage.completion_tokens

                progress.set_postfix(
                    {
                        "prompt_tokens_used": prompt_tokens_used,
                        "completion_tokens_used": completion_tokens_used,
                    }
                )

                f.write(
                    json.dumps(
                        {
                            "question": question,
                            "answer": answer,
                            "chunk": chunk,
                        }
                    )
                    + "\n"
                )
                f.flush()

# Push to the Hub.
with open(args.output) as f:
    samples = [json.loads(line) for line in f]
    dataset = Dataset.from_list(samples)
    dataset = dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
    dataset.push_to_hub(args.repo)
    print(f"Dataset pushed to: https://huggingface.co/datasets/{args.repo}")
