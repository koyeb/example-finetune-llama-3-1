# Notes

## Fine-tuning

Either we ask users to deploy a Jupyter Notebook on Koyeb and provide the step-by-step code to fine-tune the model (using LLaMa Recipes or HuggingFace Trainer) or we build a Docker image with LLaMa Recipes and run a bash script as the entry point to fine-tune the model.

Jupyter Notebook is the more traditional approach:

- ✅ Iterative, easy to debug or experiment with. Lets the user generate with the model freely before and after fine-tuning.
- ✅ No need to build a Docker image.

However, it has some downsides such as:

- ❌ Can't "One-click deploy" a notebook.

## Dataset construction

Actually there are very few pages of proper documentation for MLX. There's a ton of API reference pages for the various classes and methods though.

I fine-tuned a few LLaMas and did multiple runs on the OpenAI API and the results were not amazing. The model's able to answer about MLX, discuss some functions and classes, but it's sometimes hallucinates method parameters or the like.

Thinking of switching libraries for something "easier" to work with. Initially, I thought the Koyeb docs were a cool idea since it's larger, and fits the QA style better.

## Koyeb Agent

A fine-tuned function calling model which can perform actions on Koyeb's API.
