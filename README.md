# TellMeWhy Fine-tuning Project

## Overview
This project explores fine-tuning GPT-3.5-turbo for answering "why" questions about everyday narratives using the TellMeWhy dataset. The goal is to improve the model's ability to provide logical explanations for events and actions in short stories through different fine-tuning approaches.

## Original Source
- Dataset: https://huggingface.co/datasets/StonyBrookNLP/tellmewhy
- Base Model: GPT-3.5-turbo via OpenAI API

## Modified Files
- `354_final.ipynb`: Main project notebook containing all implementations
  - `prepare_training_data()`: Basic fine-tuning data preparation function
  - `prepare_training_data_COT()`: Chain of thought data preparation function
  - `evaluate_models()`: Evaluation metrics implementation
  - `plot_rouge_comparison()`: Visualization of results

## Commands

1. Data Preparation:
```python
# For basic fine-tuning (1000 samples)
prepare_training_data(train_df)

# For chain-of-thought fine-tuning (50 samples)
prepare_training_data_COT(train_df, 50)
```

2. Fine-tuning:
```python
# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key="your-api-key")

# Run fine-tuning
fine_tune_model(path="path-to-jsonl", model="gpt-3.5-turbo")
```

3. Evaluation:
```python
# Compare model performances
evaluate_models(client, test_df, models={
    'Base Model': 'gpt-3.5-turbo',
    'Fine-tuned Model': 'your-model-id'
})
```

## Trained Models and Data
- Basic fine-tuned model ID: ft:gpt-3.5-turbo-0125:personal::AaJ540hL
- Chain of thought model ID: ft:gpt-3.5-turbo-0125:personal::AaXOCuBB
- Training data: Available in the TellMeWhy dataset from Hugging Face
- Generated JSONL files:
  - tellmewhy_training.jsonl (1000 samples)
  - tellmewhy_training_iti_2.jsonl (50 samples with chain-of-thought prompting)

## Prompts
Basic fine-tuning prompt template:
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that explains why events happen in stories."},
        {"role": "user", "content": "Story: {narrative}\nQuestion: {question}"},
        {"role": "assistant", "content": "{answer}"}
    ]
}
```

Chain of thought prompt template:
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant trained to analyze stories and answer 'why' questions. Your goal is to explain the motivations behind events in the story by reasoning step-by-step about the context and actions described."},
        {"role": "user", "content": "Story: {narrative}\nTo answer the question, follow these steps:\n1. Summarize the key events in the story relevant to the question.\n2. Identify any actions, decisions, or events tied to the question.\n3. Think about the characters' motivations, goals, or circumstances.\n4. Combine these insights to form a clear explanation.\nQuestion: {question}"},
        {"role": "assistant", "content": "{answer}"}
    ]
}
```

## Software Requirements
- Python 3.12.6
- openai==1.3.5
- pandas==2.1.3
- numpy==1.24.3
- matplotlib==3.8.2
- seaborn==0.13.0
- rouge_score==0.1.2
- datasets==2.14.5
- An OpenAI API key with fine-tuning access

## Results
Evaluation using ROUGE metrics shows:
- Base Model: ROUGE-1: 0.138, ROUGE-2: 0.048, ROUGE-L: 0.119
- Basic Fine-tuned: ROUGE-1: 0.225, ROUGE-2: 0.076, ROUGE-L: 0.205
- Chain of Thought: ROUGE-1: 0.287, ROUGE-2: 0.099, ROUGE-L: 0.268

Notably, the chain-of-thought approach achieved these improved results while using only 50 carefully selected training examples (balanced between answerable and non-answerable questions), compared to the 1,000 examples used in basic fine-tuning. This suggests that the structured reasoning approach led to more effective learning despite using significantly less training data.
