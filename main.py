import yaml
from tasks import classificationTask
from evaluation import evaluate_all_classifications
    
def main():
    classification_results = classificationTask(save_results=True)
    # evaluate_all_classifications()

if __name__ == "__main__":
    main()