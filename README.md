# GPT-3.5 Evaluation on HumanEval Dataset

This project evaluates OpenAI's GPT-3.5 model on a sample from the HumanEval dataset to assess its code generation capabilities. The implementation is built in a way that can easily integrate new models and datasets. Parameters such as sample size and the pass@k metric are configurable. 

## Getting Started

1. Clone the repo and install required packages:
```bash
git clone https://github.com/yourusername/yourrepositoryname.git
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your OpenAI API key.
```python
OPENAI_API_KEY= 'your-openai-api-key'`
```

3. Run the evaluation script for GPT-3.5 for specific sample size and pass@k evaluation metric. 
```bash
python evaluation_pipeline.py  --model gpt-3.5-turbo --dataset humaneval --k 1 --num_eval_problems 3 --num_samples_per_task 2
```