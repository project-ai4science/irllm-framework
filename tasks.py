import random
import numpy as np
from tqdm import tqdm
import yaml
from util import classify, identify
import pandas as pd

with open('config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except:
        print("Invalid config.yml file.")

if config["task_config"]["random_seed"] >= 0:
    random.seed(int(config["task_config"]["random_seed"]))

save_results = config["task_config"]["save_results"]

def classificationTask(save_results=save_results):
    data_name = config['task_config']['data_name']
    llm = config['llm_config']['llm']
    data = pd.read_pickle(f"./data/{data_name}.pkl")
    
    results = []

    print(f"Classifying: {data_name} with {llm}")
    for row in tqdm(data.itertuples(index=False), total=len(data)):
        try:
            paper_a_id = row.id
            paper_b_id = row.b_id
            paper_b_text = row.b_text
            paper_c_id = row.c_id
            paper_c_text = row.c_text
            research_type = row.research_type
            y_true = bool(row.y_true)
            paper_a = row.a_text

            tries = 0
            while tries < 3:
                tries += 1
                try:
                    verdict, reason = classify(paper_b_text, paper_c_text)
                    break
                except Exception as e:
                    print(f"Error classifying {paper_a_id}: {e}, trying again... ({tries})")

            results.append({
                "paper_a_id": paper_a_id,
                "method": 1,
                "research_type": research_type,
                "paper_b_id": paper_b_id,
                "paper_b_text": paper_b_text,
                "paper_c_id": paper_c_id,
                "paper_c_text": paper_c_text,
                "y_true": y_true,
                "y_pred": verdict,
                "reason": reason,
                "paperA": paper_a,
            })
        
        except Exception as e:
            print(f"Unexpected error in {row.id}: {e}")

    df_results = pd.DataFrame(results)
    
    if save_results:
        file_name = f"./output/classification-{data_name}-{llm}.csv"
        print(f"Saving results: {file_name}")
        df_results.to_csv(file_name, index=False)
        
    return df_results

def recomendationTask(save_results=save_results):
    pass

def idrIdentificationTask(save_results=save_results):
    data_name = config['task_config']['data_name']
    llm = config['llm_config']['llm']
    data = pd.read_pickle(f"./data/{data_name}.pkl")
    
    results = []
    
    print(f"Classifying: {data_name} with {llm}")
    for row in tqdm(data.itertuples(index=False), total=len(data)):
        try:
            paper_a_id = row.id
            paper_a_text = row.a_text
            y_true = bool(row.y_true)
            
            title = row.title
            abstract = row.abstract

            tries = 0
            while tries < 3:
                tries += 1
                try:
                    verdict = identify(title, abstract)
                    break
                except Exception as e:
                    print(f"Error classifying {paper_a_id}: {e}, trying again... ({tries})")

            results.append({
                "paper_a_id": paper_a_id,
                "paper_a_title": title,
                "paper_a_abstract": abstract,
                "y_true": y_true,
                "y_pred": verdict,
            })
        
        except Exception as e:
            print(f"Unexpected error in {row.id}: {e}")

    df_results = pd.DataFrame(results)
    
    if save_results:
        file_name = f"./output/idridentification-{data_name}-{llm}.csv"
        print(f"Saving results: {file_name}")
        df_results.to_csv(file_name, index=False)
        
    print(f"True: {df_results.loc[df_results.y_pred == True].shape[0]}")
    print(f"False: {df_results.loc[df_results.y_pred == False].shape[0]}")

    return df_results