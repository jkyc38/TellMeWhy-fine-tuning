# TellMeWhy-fine-tuning
## Overview
This project explores fine-tuning GPT-3.5-turbo for answering "why" questions about everyday narratives using the TellMeWhy dataset. The goal is to improve the model's ability to provide logical explanations for events and actions in short stories.

## Project Components

### Data
- Uses the StonyBrookNLP/tellmewhy dataset
- Contains narrative-question-answer triplets focusing on causal reasoning
- Dataset split into training, validation, and test sets

### Implementation
- Fine-tuning GPT-3.5-turbo using OpenAI's API
- Two approaches implemented:
  1. Basic fine-tuning with direct question-answering
  2. Chain-of-thought prompted fine-tuning for improved reasoning

### Evaluation
- Measures model performance using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- Compares baseline GPT-3.5-turbo against fine-tuned versions
- Includes visualization of performance metrics

## Results
The fine-tuned model showed significant improvements over the baseline:
- ROUGE-1: 28.72% (fine-tuned) vs 18.71% (baseline)
- ROUGE-2: 9.93% (fine-tuned) vs 7.56% (baseline)
- ROUGE-L: 26.85% (fine-tuned) vs 16.15% (baseline)

## Requirements
- Python 3.x
- OpenAI API key
- Required packages:
  - openai
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - rouge_score
  - datasets

## Usage
1. Set up your OpenAI API key
2. Load and preprocess the TellMeWhy dataset
3. Run fine-tuning scripts
4. Evaluate model performance using provided metrics

## Files
- `354_final.ipynb`: Main notebook containing all code and analysis
- Various JSON files for training data and model outputs
- Evaluation scripts and visualization tools

## Future Work
- Experiment with different prompting strategies
- Implement additional evaluation metrics
- Explore other model architectures and fine-tuning approaches

## License
[Include your chosen license here]

## Contributors
[Your name/organization]

## Acknowledgments
- StonyBrookNLP for the TellMeWhy dataset
- OpenAI for the GPT-3.5-turbo model and API
