---
layout: post
title:  "LLM Reinforcement Learning with GRPO"
excerpt: "What I've learnt from attempting post-training reinforcement learning on open weights LLM."
date:   2025-03-03
hide: false
categories: deep-learning llm visualisation rl
permalink: /2025/03/03/llm-post-train-grpo-take-aways.html/
---

## TLDR 
I've tried reinforcement learning (RL) techinique leveraged by DeepSeek on a specific training dataset to improve the reasoning capability of large language models. Though there are significant improvements over the base model, a lot of design choices and experimentation are needed for achieving effective and efficient RL as a critical post-training step in open weights LLMs towards reliably stronger reasoning capability. These choices include but are not limited to:
* Choice and format of fine-tuning dataset
* Base model to perform RL on - pretrained base model vs models with instruction supervised fined tuning (SFT) + reinforcement learning with human feedback (RLHF)
* Full fine tuning vs parameter efficient fine tuning (PEFT) methods (+ choice of finetuning hyperparameters in the case of LORA e.g. rank and alpha)
* Whether to perform further chain-of-thought (CoT) instruction SFT first before undertaking RL - DeepSeek R1 did this but not R1-zero
* Reward function(s) - extremely important in RL for specifying what we want to optimise towards
* Evaluation methodology - how to evaluate success towards reasoning capability before and after post-training 
* Fine-tuning hyperparameters such as number of generations per step, training batch size, number of steps, learning rate, learning schedule etc.

## Setup

This article closely follows this [Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) by Unsloth with some noticeable changes (details below). Credits to the [Unsloth team](https://unsloth.ai/) for their wonderful notebooks and open source approach towards finetuning LLMs!

### Hardware
This experiment is ran on a Google Colab notebook in 2 hours on A100 GPU with the number of training steps capped initially to 250 then another < 1 hour extending the number of steps to 350.

### Software

**Training Framework:** [Unsloth](https://unsloth.ai/) is a framework built on top of Huggingface Transformers library, that allows for significant boost in speed and reduction in GPU memory needed when fine-tuning most of the major open weights LLMs. 

**Reinforcement Learning:** Huggingface [TRL](https://huggingface.co/docs/trl/en/index) library which is been leveraged for Group Relative Policy Optimisation (GRPO). GRPO is the RL algorithm used when training DeepSeek-R1-Zero and DeepSeek-R1. 

**Inference Framework:** [VLLM](https://docs.vllm.ai/en/latest/) library allows for high-throughput efficient inference of LLMs. It's also been integrated into Unsloth.

**Training Monitoring:** [Weghts and Biases](https://wandb.ai/site/) library to visualise model's performance against many of the internal hyperparameters and metrics such as learning rates, reward values, loss etc. It's a great platform for surfacing and monitoring metrics and logs during the model training / fine-tuning processes. It's been integrated into Transformers and Unsloth.

**Dataset:** Huggingface [datasets](https://huggingface.co/docs/datasets/en/index) library contains a vast amount of publicly available datasets. 

**Visualisation:** Google's [Embedding Projector](https://projector.tensorflow.org/) to visualise the text dataset by their embeddings and other metadata.  

## Methodology

### Base Model
The base LLM chosen was `meta-llama/meta-Llama-3.1-8B-Instruct` from in line with the Google Colab example in [Unsloth blog](https://unsloth.ai/blog/r1-reasoning). In hindsight, we can experiment with: 
* Models like as this one that already been through post-training stages of the standard instruction following SFT and RLHF towards better alignment with human conversatioal styles, formats and preferences in text generation OR
* Directly performing RL on base models `meta-llama/meta-Llama-3.1-8B` - this may come at a risk of non-human-interpretable chain of thougt (refer to DeepSeek-R1-Zero's appoach) OR
* Using base models `meta-llama/meta-Llama-3.1-8B` followed by your own CoT SFT (CoT datasets can be found through Huggingface Datasets) before performing RL then followed by standard SFT and RLHF (refer to DeepSeek-R1's approach)

### Dataset
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) which stands for "Grade School Math 8K" is the mathematical reasoning dataset used here to fine-tune towards a better mathematical reasoning model. 
> "The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning." - [Dataset Card for GSM8K](https://huggingface.co/datasets/openai/gsm8k)  

See an example below on the types of questions and answers.

| Question | Answer |
| ----------- | ----------- |
| Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients? | A pound of cream cheese cost $10 / 2 = $<<10/2=5>>5. A pack of cold cuts cost $10 x 2 = $<<10*2=20>>20. Jasper spent $10 + $5 + $20 = $<<10+5+20=35>>35 on the ingredients. #### 35 |  

### EDA
To get a sense of the underlying fine-tuning dataset, I've created embeddings and visualised them below.  
[![embeddings-gif](/assets/grpo/gsm8k-embeddings-viz.gif)](/assets/grpo/gsm8k-embeddings-viz.gif "GSM8K Embeddings"){:height="100%" width="100%"}  
Few preliminary take-aways below: 
* Train (in blue) and test (in red) set splits are roughly randomised in the sematic embedding space
* There's an obvious small cluster of points that talks about age, years, family etc.
* Semantic similarity in the embedding space gives no indication of the varying levels of difficulty in the tasks - perhaps we can use the number of steps taken in the solution part of the datset (see the steps taken in the "Answer" part of the table above) as a proxy

### Training Set 
#### System Prompt
The system prompt in the original Google Colab as below.
```
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
```
From the original Unsloth notebook, I've added another line (see below) at the end to leverage the model's instruction following capability. I found this extra line to be especially effectively at lifting the performance of the pretrained `meta-llama/meta-Llama-3.1-8B-Instruct` before any reasoning RL post-training is performed. Even though this extra prompt line may discount the increase in performance of reasoning RL, I feel it's a fairer approach when measuring performance against the dataset. 
```
You must put your reasoning between the <reasoning> tags and answer between the <answer> tags.
```

#### Chat Template
The final prompt passed into the instruct model is of the following chat template with a system prompt at the start followed by the user message which is just the **Question** column of the GSM8K dataset. The final answer is the text extracted after the final #### keyphrase in the **Answer** column in the table above. 
```
[
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': x['question']}
]
```

### Evaluation
When evaluating the generated answers against gold label answers in the GSM8K dataset, we can do it in many ways.
#### Hard Correctness
The text between the answer tags of the generated answer must equal exactly to the gold label. This is the original code from the Unsloth notebook. If there are no <answer> tags or missing tags then the entire length of text would be extracted up to finding the tags.
```
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()
```

#### Soft Correctness
I've added a less strict version of the corretness here by extracting the first number from the text between the answer tags. This is so that I can track the capability of the model at a softer level and see this number improve at a faster rate e.g. if the answer is in the format of '96%' or '72 units' or '$27' and gold labels are '96', '72' or '27' it'll be marked correct under this measure.
```
def extract_final_answer(text: str) -> str:
    answer = extract_xml_answer(text)
    pattern = r'[$]?(\d+(?:,\d+)*)%?'
    matched = re.search(r'[$]?(\d+(?:,\d+)*)%?', answer)
    if matched:
        return matched.group(1).replace(', ', '').replace(',', '')
    return '0' # if not found extract '0' as final answer
```

### Test Set
The test set is constructed in the same way as the training set but with the "test" split of the GSM8K dataset. What I started to realise is that this dataset may not be challenging enough as the official `Llama-3.1-8B-Instruct` model performance reached [84.5](https://ai.meta.com/blog/meta-llama-3-1/) with 7-shot Chain-of-Thought (COT) reasoning on GSM8K. With 5-shot COT we're seeing a drop to [82.3](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/81).  

I've decided to sample 50% from the GSM8K test set and saw around 70% as the accuracy (with soft correctness) after running multiple times. Below is the simple evaluation function for matching predicted with ground truth labels. This evaluation can be applied on the extracted answers from either hard correctness or soft correctness meaures.

```
def evaluate_answers(trues:list, preds:list):
    assert len(preds) == len(trues)
    return sum([1 if t == p else 0 for t, p in zip(trues, preds)]) / len(trues)
```

### Reward Functions
For RL there are many rewards we need to manually define for the model to optimise for.  

**Correctness** The hard correctness mentioned above and implemented as `correctness_reward_func` in the notebook. I've dialled the reward value down from 2.0 to 1.0 as I added another correctness measure below.  

**Soft Correctness** The soft correctness mentioned above and I've added this as `soft_correctness_reward_func`. We get 1.0 if the conditon is met and 0.0 otherwise.
```
def soft_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_final_answer(r) for r in responses]
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
```  

**Integer** Rewards 0.5 for if the final answer between the answer tags can be converted to an integer and implemented as `int_reward_func` in the blog.  

**Strict Format** Rewards 0.5 for if the final answer follows a strict format of both open and close html tags for reasoning and answer sections. Implemented as `strict_format_reward_func` in the Unsloth notebook but I've changed this a little to allow for zero or more new lines at the end and multiple lines in the regex.
```
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n*$" # ADDED: Zero or more new lines at end
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]  # ADDED re.DOTALL for new lines
    return [0.5 if match else 0.0 for match in matches]
```  

**Soft Format** Very similar to the strict format version but more lenient with where these tags can be located in the generated text. Implemented as `soft_format_reward_func` in the blog. Again, I've added `re.DOTALL` when matching using regex for new lines.  

**XML Tags Count** This funciton rewards 0.125 for each of the 4 tags and penalises for long answers between the answer tags. Implemented as `xmlcount_reward_func` in the blog.

I've also added another function to outputs the rewards given the test set and output text generated by the model.
```
def evaluate_rewards(dataset_test_sample, output_texts:list[text]):

    completions = [[{"content" : text}] for text in output_texts]

    correctness = correctness_reward_func(
        dataset_test_sample['prompt'], 
        completions, 
        answer=dataset_test_sample['answer']
    )

    soft_correctness = soft_correctness_reward_func(
        dataset_test_sample['prompt'], 
        completions, 
        answer=dataset_test_sample['answer']
    )
        
    int_rewards = int_reward_func(completions)
    soft_format_rewards = soft_format_reward_func(completions)
    strict_format_rewards = strict_format_reward_func(completions)

    print('Correctness', round(sum(correctness) / len(correctness), 3))
    print('Soft correctness', round(sum(soft_correctness) / len(soft_correctness), 3))
    print('Integer', round(sum(int_rewards) / len(int_rewards), 3))
    print('Soft format', round(sum(soft_format_rewards) / len(soft_format_rewards), 3))
    print('Strict format', round(sum(strict_format_rewards) / len(strict_format_rewards), 3))
```

### Model Training
The default training hyperparameters per the Unsloth notebook are used except for:
* `gradient_accumulation_steps = 2` instead of 1 for smoother training
* `num_generations = 8` instead of 6 for greater variations in each generation so it'll be more likely to hit higher rewards

## Results

### Pretrained Model Evaluation
Here I've added a part to evaluate the base model before going through RL. For this I've used the same system prompt and user message that's used in training and later used for post-training evaluation. Two callouts below: 
* Pretrained model performs at 70% when using a less strict method for measuring correctness
* There's a lot of room for improvement in hard correctness and integer reward

| Metric Type | Value | Max Value |
| ----------- | ----------- | ----------- |
| Hard correctness | 0.287 | 1.0
| Soft correctness | 0.701 | 1.0
| Integer reward | 0.164 | 0.5
| Soft format reward |  0.442 | 0.5
| Hard format reward | 0.421 | 0.5


### Model Training Metrics
See the numbers below for training numbers logged in Weights and Biases run.
[![](/assets/grpo/grpo-wandb-train-1.jpg)](/assets/grpo/grpo-wandb-train-1.jpg "GRPO Training Metrics 1"){:height="100%" width="100%"}  
[![](/assets/grpo/grpo-wandb-train-2.jpg)](/assets/grpo/grpo-wandb-train-2.jpg "GRPO Training Metrics 2"){:height="100%" width="100%"}

### Post-RL Model Evaluation
After 300 steps, the model evaluation on the unseen test dataset is as below. Note that the numbers after the entire 350 steps are slightly worse than the ones below but still showed significant improvement over the original pretrained model evaluation numbers above. This may be due to the model still trying to learn the training dataset as it only observed a small fraction (< 10% of an epoch)  

| Metric Type | Value | Max Value |
| ----------- | ----------- | ----------- |
| Hard correctness | 0.785 | 1.0
| Soft correctness | 0.789 | 1.0
| Integer reward | 0.469 | 0.5
| Soft format reward |  0.490 | 0.5
| Hard format reward | 0.489 |  0.5

Results show a good amount of improvement especially in hard correctness and integer rewards. It is intuitive that we see matching increase in integer reward when hard correctness go up as the hard correctness by definition requires answers to be integers as a precondition (refer to the final answers in the GSM8K dataset). Snippets below show examples of how the model behaves towards the end of the training process as it almost always output integers (though it can sommetimes still be wrong). 

```
-------------------- Question:
Ursula earns $8.50 an hour working in a restaurant. She works 8 hours a day. If she works 20 days a month, determine her annual salary. 
Answer:
16320 
Response:
<reasoning>
To find Ursula's annual salary, first, we need to calculate her daily income, then her monthly income, and finally her yearly income. 
Daily income is 8.50 * 8 = 68, monthly income is 68 * 20 = 1360,  annually her income would come out as 1360 * 12 = 16320 
</reasoning>
<answer>
16320
</answer> 
Extracted:
16320
-------------------- Question:
Carla is dividing up insurance claims among 3 agents. Missy can handle 15 more claims than John, who can handle 30% more claims than Jan. If Jan can handle 20 claims, how many claims can Missy handle? 
Answer:
41 
Response:
<reasoning>
Jan's claims = 20. John's claims = 20 * 1.3 = 26. Missy's claims = 26 + 15 = 41.
</reasoning>
<answer>
41
</answer> 
Extracted:
41
-------------------- Question:
Viggo spent $80 on a shirt. He paid the shirt using $20 and $10 bills. If she gave the clerk one more $20 bill than $10 bills, how many $10 bills did he give? 
Answer:
2 
Response:
<reasoning>
Let x be the number of $10 bills given. Since Viggo gave one more $20 bill than $10 bills, the number of $20 bills is x + 1. The total amount paid is 0.10x + 20(x + 1) = 80. Simplifying this, you get 0.10x + 20x + 20 = 80. Combining like terms gives 20.10x + 20 = 80. Subtract 20 from both sides to get 20.10x = 60, then divide by 20.1 to get x = 3.
</reasoning>
<answer>
3
</answer> 
Extracted:
3
```

## Next Steps
For the curious minds, there are many more experiments we can do with this setup to understand further about reasoning models. At a high level we can try:
* **Datasets** - more challenging ones e.g. the [Huggingface cookbook on GRPO](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) trains on this competition level high school maths and maths Olympiad dataset called ["NuminaMath"](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR)
* **Models and model types** - especially base pretrained models that hasn't been through SFT and RLHF e.g. `meta-llama/Llama-3.1-8B` and observe for yourself if the model optimises towards non-human-readable sequence of tokens (referenced in DeepSeek-R1-Zero).
* **Model sizes** - see how the performance scales with base model sizes
* **Reward functions** - get creative about what you value as important for the model to learn
* **RL algorithms and setups** - this is going to need the most amount of changes as we diverge from GRPO and look at other ways to perform test-time-scaling with RL. One reference article by Huggingface [here](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) for obtaining rewards during intermediate generations instead of the at the end.