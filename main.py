import yaml
from tasks import classificationTask, recomendationTask, idrIdentificationTask
from evaluation import evaluate_all_classifications
    
with open('config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except:
        print("Invalid config.yml file.")



def main():
    task = config["task_config"]["task"]
    results = None
    
    if task == "classification":
        results = classificationTask()
    elif task == "recomendation":
        results = recomendationTask()
    elif task == "idridentification":
        results = idrIdentificationTask()
    elif task == "evaluation":
        evaluate_all_classifications()
    else:
        print("Invalid task.")

if __name__ == "__main__":
    main()