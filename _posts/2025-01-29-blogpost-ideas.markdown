---
layout: post
title:  "Ideas around next blog bost"
excerpt: "Jot down of ideas as candidates for the next blog posts"
date:   2025-01-29 
hide: false
categories: deep-learning llm visualisation
permalink: /2025/01/29/blogpost-ideas.html/
---

In this post I will lay out some ideas that can be used for future blog posts around using Small Language Models (SLMs). In my current definition, an SLM is a language model that can be run on a laptop, with a maximum number of parameters of around 14B, which can be quantised at most to `int-4` (beyond which the generation qualitiy deteriorates drastically).

## Model Performance Measurement
There are different layers when measuring performance of Langauge Models at various tasks especially in the age of LLMs. You'll need a dataset and a language model as input. The first three levels can be further broken down by the specific sub-groups of your dataset.
1. Discrete level - based on the token outputs of your models
2. Continuous level - based on logits / probabilities of your model output
3. Model internals - based on embeddings / activations of various layers within your model
4. Further processing of your model internals
5. Other interpretability methods - SHAP, LIME, Text Saliency methods etc.

## Mechanistic Interpretability
This relates to point 4 above

## Visualisation Tools 
This relates to all points within model performance measurement
