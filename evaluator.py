import os
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from abc import ABC, abstractmethod

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, samples_file, problems_eval):
        pass

class HumanEvalEvaluator(Evaluator):
    def __init__(self, k, num_eval_problems= 2, num_samples_per_task =2, n_workers=4, timeout=3.0, ):
        self.dataset = 'data/HumanEval.jsonl.gz'
        self.k = k
        self.n_workers = n_workers
        self.timeout = timeout
        self.num_eval_problems = num_eval_problems
        self.num_samples_per_task =num_samples_per_task

    def sample_problems(self, generator):
        problems = read_problems(self.dataset)
        problems_eval = dict(list(problems.items())[:self.num_eval_problems])
        samples = [
            {
                "task_id": task_id,
                "completion": generator.generate_one_completion(problems[task_id]["prompt"])
            }
            for task_id in problems_eval
            for _ in range(self.num_samples_per_task)
        ]
        return samples, problems_eval

    def execute_evaluation(self, generator):
        if os.path.exists(self.dataset):
            samples, problems_eval = self.sample_problems(generator)

            output_file_path = "results/samples.jsonl"
            write_jsonl(output_file_path, samples)

            result = self.evaluate(output_file_path, problems_eval)
            print(result)
        else:
            raise FileNotFoundError(f"The dataset {self.dataset} does not exist.")

    def evaluate(self, samples_file, problems_eval):
        return evaluate_functional_correctness(samples_file, k=[self.k], problems=problems_eval, n_workers=self.n_workers, timeout=self.timeout, problem_file=self.dataset)
