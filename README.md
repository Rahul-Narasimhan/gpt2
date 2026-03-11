# GPT-2 Reproduction on Wiki 2M Tokens

A from-scratch reproduction of a GPT-2 style decoder-only transformer in PyTorch, trained on a ~2 million token Wikipedia-based dataset.

## Overview

This project is my hands-on implementation of a GPT-2 style language model using PyTorch. Instead of using a high-level pretrained model API, I implemented the core components manually to understand how autoregressive transformers work in practice, including model architecture, training, validation, and text generation.

The project currently contains separate scripts for model definition, training, and sampling:

- `model.py` — model architecture
- `train.py` — training loop
- `sample.py` — text generation / inference

## Objectives

The goals of this project were:

- understand decoder-only transformers from first principles
- implement GPT-style autoregressive language modeling manually
- train on a small tokenized corpus
- track training and validation loss
- generate text from the trained model
- build a clean, explainable project that reflects real understanding rather than only library usage

## Implemented Components

This project includes the following core components:

- token embeddings
- positional embeddings
- multi-head causal self-attention
- feed-forward MLP block
- residual connections
- layer normalization
- autoregressive next-token prediction
- training loop in PyTorch
- validation loss estimation
- text generation from a trained model

## Repository Structure

```text
gpt2_wiki_2m/
│
├── gpt2_model.py      # GPT-2 style model architecture
├── train.py      # training script
├── output_generation.py     # text generation script
├── README.md
├── requirements.txt
└── .gitignore

Tech Stack

Python
PyTorch
NumPy
tiktoken

How to Run
1. Clone the repository

git clone https://github.com/Rahul-Narasimhan/gpt2-wiki-2m.git
cd gpt2-wiki-2m

2. Install dependencies
pip install -r requirements.txt

3. Train the model
python train.py

4. Generate text samples
python sample.py

Training Setup

This model is being trained on a small Wikipedia-based dataset of roughly 2 million tokens as part of a learning-focused GPT-2 reproduction effort.

Current setup includes:

decoder-only transformer
next-token prediction objective
train/validation split
validation loss tracking during training
generation using the trained model

You can update this section later with exact hyperparameters such as:

number of layers
number of heads
embedding dimension
block size
batch size
optimizer
learning rate
total training steps
hardware used

Results

This section will be updated with training outputs after runs are completed.

Planned additions:

final training loss

final validation loss

loss curve plots

training configuration summary

example generated samples

Why I Built This

I built this project to understand GPT-style autoregressive transformers beyond high-level APIs. Instead of relying only on prebuilt abstractions, I wanted to implement the architecture, training loop, and sampling pipeline manually in PyTorch so I could understand what each part is doing and how the full system fits together.

This project is part of my broader effort to develop deeper practical understanding of large language models, training dynamics, and transformer internals.

What I Learned

Through this project, I am working to strengthen my understanding of:

how token and positional embeddings interact

how causal self-attention works

why residual connections and layer normalization matter

how logits are produced for next-token prediction

how training and validation loops are structured

how sampling works at inference time

how to move from conceptual understanding to working PyTorch implementation

Next Steps

add checkpoint saving and loading

improve logging and experiment tracking

add training and validation loss curves to the README

make hyperparameters configurable

train on a larger dataset

compare outputs across different training runs

improve generation quality and sampling controls

Notes

This is a learning-driven implementation intended to build strong first-principles understanding of GPT-style language models. The goal is not to claim a production-scale GPT-2 reproduction, but to implement the main components honestly and understand them deeply.

License
