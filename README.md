## Training and Evaluation Scripts for Urdu LLM Finetuning

Configured for testing of Llama model family and Cohere Aya (multilingual) in 8B param size.
- Llama models will include 3.2 3B (multlingual), Llama 3.1 8B (not multilingual), and potentially 3.2 1B to explore relative improvement in-family from multilingual pretraining.

data_formatting.ipynb cleans up the existing corpus for training.
Training is in progress using TACC (Texas Advanced Computing Cluster), and environment setup is handled there via conda. Slurm job scripts will be included here once validated.

### TODOs:
- implement logic for Llama evaluation (should look similar to Aya)
- implement translation testing code and gather corpus for those experiments


### Project overview

The goal of this project is to explore the latent properties of LLMs (both multilingual pretraining and not) after finetuning on Urdu language datasets, as an example of a low-resource language in NLP. There are two goals:
- probing Urdu text generation / next token prediction before and after finetuning
- testing model performance on translation to create a small translation model for Urdu

Within the first includes investigating the linguistic properties of the learned representations after finetuning: are Urdu-specific features learned in the embeddings or activations that can distinguish it from Hindi, Farsi, and other related languages? Would performance improve significantly with language or text-specific tokenziation?
- creating a custom Hindi-Urdu dataset containing synonym pairs derived from (Sanskrit, Farsi/Arabic)
- investigating SAEs and interpretability methods laid out in recent works such as [this](https://arxiv.org/pdf/2505.20546) and [this](https://arxiv.org/pdf/2506.00653) to probe these models

On the translation side, the goal will be to investigate performance and potentially the impacts of quantization, in pursuit of the smallest high-quality translation model for colloquial and eventually poetic/literary Urdu text.