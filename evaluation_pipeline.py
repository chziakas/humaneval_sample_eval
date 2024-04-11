import argparse
from completion_generator import OpenAICompletionGenerator
from evaluator import HumanEvalEvaluator

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate completions using OpenAI's API.")
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to be used for generating completions.')
    parser.add_argument('--dataset', type=str, default='humaneval', help='Dataset for evaluation.')
    parser.add_argument('--k', type=int, default=1, help='k for Pass@K metric.')
    parser.add_argument('--num_eval_problems', type=int, default=2, help='Number of evaluation problems to process.')
    parser.add_argument('--num_samples_per_task', type=int, default=2, help='Number of samples per task to generate.')
    args = parser.parse_args()

    if args.model == 'gpt-3.5-turbo':
        generator = OpenAICompletionGenerator(args.model)
    else:
        raise ValueError("Currently only gpt-3.5-turbo is supported.")
    
    if args.dataset.lower() == 'humaneval':
        evaluator = HumanEvalEvaluator(k=args.k, num_eval_problems=args.num_eval_problems, num_samples_per_task=args.num_samples_per_task)
        # Start the evaluation process 
        evaluator.execute_evaluation(generator)
    else: 
        raise ValueError("Currently only HumanEval is supported.")

if __name__ == '__main__':
    main()