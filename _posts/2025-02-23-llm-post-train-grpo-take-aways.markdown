---
layout: post
title:  "LLM Reinforcement Learning with GRPO"
excerpt: "What I've learnt from attempting post-training reinforcement learning on open weights LLM."
date:   2025-02-23
hide: true
categories: deep-learning llm visualisation rl
permalink: /2025/02/23/llm-post-train-grpo-take-aways.html/
---

TLDR: A lot of design choices and experimentation are still needed for effective and efficient Reinforcement Learning as a post-training step in open weights LLMs.

## Setup

### Hardware
I ran this experiment on a Google Colab notebook. The number of steps is capped to 250.

### Software

#### Training Framework
[Unsloth](https://unsloth.ai/) is a framework built on top of Huggingface Transformers library, that allows for significantly improved speed and less memory needed when fine-tuning most of the major open weights LLMs. I've followed closely their [blog](https://unsloth.ai/blog/r1-reasoning) on fine-tuning for reasoning models.

#### Reinforcement Learning
Huggingface [TRL](https://huggingface.co/docs/trl/en/index) library which is been leveraged for GRPO reinforcement learning. Group Relative Policy Optimisation is the RL methodology used when training DeepSeek-R1-Zero and DeepSeek-R1. 

#### Dataset
Hugginface [datasets](https://huggingface.co/docs/datasets/en/index) library which contains a vast amount of publicly available datasets. [GSM8K](https://huggingface.co/datasets/openai/gsm8k) which stands for "Grade School Math 8K" is the mathematical reasoning dataset used here to fine-tune towards a better reasoning model. "The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning." See example below.

| Question | Answer |
| ----------- | ----------- |
| Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients? | A pound of cream cheese cost $10 / 2 = $<<10/2=5>>5. A pack of cold cuts cost $10 x 2 = $<<10*2=20>>20. Jasper spent $10 + $5 + $20 = $<<10+5+20=35>>35 on the ingredients. #### 35 |

#### Visualisation
I've used Google's [Embedding Project](https://projector.tensorflow.org/) to visualise the GSM8K dataset.
![embeddings-gif](/assets/grpo/gsm8k-embeddings-viz.gif){:height="100%" width="100%"}

#### Training Monitoring
To visualise model's performance against many of the internal hyperparameters and metrics such as learning rates, reward values, loss etc. I used [Weghts and Biases](https://wandb.ai/site/). It's a great platform for surfacing and monitoring metrics, logs and many more during model training processes.

## Methodology

### Model Loading
The LLM used is `meta-llama/meta-Llama-3.1-8B-Instruct`.

## Results

### Model Metrics
See the numbers below for Weights and Biases training run.
![]({{ "/assets/grpo/grpo-wandb-train-1.jpg" | absolute_url }}){:height="100%" width="100%"}
