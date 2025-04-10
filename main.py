import argparse
from tasks import TaskHandler
from evaluation import Evaluator
from utils import load_config

config_path = './config.yml'
CONFIG = load_config(config_path)

def main():
    task_config = CONFIG['task_config']
    assert args.task in task_config['task_list']
    
    # map the args to actual task
    if args.task == "output_evaluation":
        eval_type = args.eval_type
        evaluator = Evaluator()
    else:
        # task_handler = TaskHandler(provider=args.provider, model_name=args.model_name, lm_config_path=config_path, save_path=save_path, **task_config)
        task_handler = TaskHandler(provider=args.provider, model_name=args.model_name, lm_config_path=config_path, **task_config)
        task_func = task_handler[args.task]
        task_func(file_names=["data_exp_2_3.json"], verbose=True)
        # task_func(verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="main python file to do experiments and result evaluation"
    )
    parser.add_argument("--provider", type=str, help="Provider name (gpt, llama, gemini, grok, deepseek)", default="gpt")
    parser.add_argument("--model_name", type=str, help="Model name", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--task", type=str, help="Task type (e.g., classification, recommendation, generation)", required=True)
    parser.add_argument("--eval_type", type=str, help="Choose the type of result evalutaion for different tasks", default=None)

    # parser.add_argument("--save_path", type=str, help="subfolder to save the experiment results", default=None)
    # parser.add_argument("--out_path", type=str, help="subfolder to save the evaluation results", default=None)
    args = parser.parse_args()
    main()