# import random
import os
import re
import numpy as np
from tqdm import tqdm
from lm_client import LM_Client
from utils import *
# from evaluation_v1 import calculate_mrr, ndcg_at_10
import pandas as pd
from typing import *
from copy import deepcopy
import random
import json


class TaskHandler():

    def __init__(self, provider: str, model_name: str, save_path: str, lm_config_path: str, data_path: str = './data', **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.lm_config_path = lm_config_path
        self.save_path = save_path
        self.data_path = data_path
        self.task_mapping = {
            "identification": self.identify_task, # exp_1
            "classification": self.classify_task, # exp_2.1 + exp_2.2 + exp_2.3
            "recommendation": self.recommend_task, # exp_3.1 + exp_3.2
            "recommendation_ranking": self.recommend_ranking_task, # exp_3.1 + exp_3.2
        }
        self.client = LM_Client(provider=self.provider, model_name=self.model_name, config_path=self.lm_config_path)
        self.kwargs = kwargs

    def __getitem__(self, key: Literal["classification", "identification", "recommendation"]):
        """Enables retrieval using Object[key]."""
        return self.task_mapping.get(key, "No matching tasks identified")
    
    # exp_1
    def identify_task(self, file_name: str = "data_exp_1.json", verbose: bool = False, checkpoint_len: int = 5):
        # check if need budget:
        budget = self.kwargs.get("budget_mode", None)
        budget_num = self.kwargs.get("budget_num", None)
        # check if we want a critical llm
        critical = self.kwargs.get("critical", False)
        # assemble file name
        out_file_name = f"exp_1_{self.model_name}"
        if budget:
            out_file_name = f"exp_1_{self.model_name}_budget"
        if critical:
            out_file_name = f"{out_file_name}_critical"
        out_file_name += ".json"
        # checkpoint system to obtain start point
        cached_idx = 0
        cached = False
        # check if the file exists and load it
        if os.path.exists(os.path.join(self.save_path, out_file_name)):
            df_cached = pd.read_json(os.path.join(self.save_path, out_file_name))
            cached_idx = df_cached.shape[0]
            cached = True
        if verbose:
            print(f"Already processed {cached_idx} samples. Continuing from there...")
            print(f"Doing data file: {file_name}...")
            print(f"Budget mode: {budget}, num of yes to say: {budget_num}")
            print(f"Critical llm: {critical}")
        # load benchmark data
        df = pd.read_json('/'.join([self.data_path, file_name]), dtype={'id': str, 'date': str})#[:5] # first try 5 samples to ensure works well
        df_size = df.shape[0] - cached_idx
        # break the function if no data
        if df_size == 0:
            if verbose:
                print(f"No new data to process in {file_name}.")
            return
        # slice the dataframe to start from the last processed index
        df = df[cached_idx:]
        ids, responses, labels, verb_conf, response_logprobs = [], [], [], [], []
        if cached:
            # load the cached data
            ids = df_cached['id'].tolist()
            responses = df_cached['y_pred'].tolist()
            labels = df_cached['y_true'].tolist()
            verb_conf = df_cached['verb_conf'].tolist()
            response_logprobs = df_cached['log_probs'].tolist()
        # main loop
        for i, each in tqdm(df.iterrows(), total=df_size):
            title, abstract = each['title'], each['abstract']
            # budget system
            if budget:
                remaining_budget = budget_num - sum(responses)
                if remaining_budget == 0:
                    break
                input_prompt = prompt_exp_1_budget % (remaining_budget, title, abstract)
            else:
                input_prompt = prompt_exp_1 % (title, abstract)
            # llm prompt generation
            response_txt, logprobs = self.client.generate(sys_prompt=sys_prompt_critical if critical else sys_prompt, input_prompt=input_prompt)
            # capture groups for verdict and reason.
            verdict, verb_score = None, None
            pattern = r"Your verdict:\s*(?P<verdict>Yes|No).?\s*Confidence score:\s*(?P<score>\d+).?"
            match = re.search(pattern, response_txt, re.IGNORECASE)
            if match:
                data = match.groupdict()
                # Map verdict values to boolean
                verdict_mapping = {"yes": True, "no": False}
                # Convert verdict to lowercase to match our mapping keys
                data["verdict"] = verdict_mapping.get(data["verdict"].lower())
                # add to the values
                verdict, verb_score = data["verdict"], int(data["score"])
            else:
                if verbose:
                    print(f"Match not found. Response: {response_txt}")
                continue
            # update the result collection
            ids.append(each['id'])
            verb_conf.append(verb_score)
            responses.append(verdict)
            response_logprobs.append(logprobs)
            labels.append(each['y_true'])
            # save checkpoint
            if i % checkpoint_len == 0:
                if verbose:
                    print(f"Making checkpoint on file: {file_name}...")
                # collect result and put into the df
                data = {
                    "id": ids,
                    "y_true": labels,
                    "y_pred": responses,
                    "verb_conf": verb_conf,
                    "log_probs": response_logprobs
                }
                
                # save to json file
                pd.DataFrame(data).to_json('/'.join([self.save_path, out_file_name]), indent=2, index=False, orient='records')

        # collect result and put into the df
        data = {
            "id": ids,
            "y_true": labels,
            "y_pred": responses,
            "verb_conf": verb_conf,
            "log_probs": response_logprobs
        }
        # save to json file
        pd.DataFrame(data).to_json('/'.join([self.save_path, out_file_name]), indent=2, index=False, orient='records')

    # exp_2
    def classify_task(self, file_names: list = [f"data_exp_2_{i+1}.json" for i in range(3)], verbose: bool = False, checkpoint_len: int = 5): 
        # check if need budget:
        budget = self.kwargs.get("budget_mode", None)
        budget_num = self.kwargs.get("budget_num", None)
        # check if we want a critical llm
        critical = self.kwargs.get("critical", False)
        if verbose:
            print(f"Budget mode: {budget}, num of yes to say: {budget_num}")
            print(f"Critical llm: {critical}")
        for idx, file_name in enumerate(file_names):
            out_file_name = f"exp_2_{idx+1}_{self.model_name}"
            # assemble out file name
            if budget:
                out_file_name += "_budget"
            if critical:
                out_file_name += "_critical"
            out_file_name += ".json"
            # checkpoint system to obtain start point
            cached_idx = 0
            cached = False
            # check if the file exists and load it
            if os.path.exists(os.path.join(self.save_path, out_file_name)):
                df_cached = pd.read_json(os.path.join(self.save_path, out_file_name))
                cached_idx = df_cached.shape[0]
                cached = True
            if verbose:
                print(f"Already processed {cached_idx} samples. Continuing from there...")
                print(f"Doing data file: {file_name}...")
            # load benchmark data
            df = pd.read_json(os.path.join(self.data_path, file_name))#[:5] # first try 5 samples to ensure works well
            df_size = df.shape[0] - cached_idx
            # break the function if no data
            if df_size == 0:
                if verbose:
                    print(f"No new data to process in {file_name}.")
                return
            # slice the dataframe to start from the last processed index
            df = df[cached_idx:]
            ids, responses, labels, reasons_pred, verb_conf, response_logprobs = [], [], [], [], [], []
            if cached:
                # load the cached data
                ids = df_cached['id'].tolist()
                responses = df_cached['y_pred'].tolist()
                labels = df_cached['y_true'].tolist()
                reasons_pred = df_cached['reasons'].tolist()
                verb_conf = df_cached['verb_conf'].tolist()
                response_logprobs = df_cached['log_probs'].tolist()
            # main loop
            for i, each in tqdm(df.iterrows(), total=df_size):
                disci_one = ["Title: %s; Abstract: %s" %(title, abstract) for title, abstract in zip(each['b_title'], each['b_abstract'])]
                disci_two = ["Title: %s; Abstract: %s" %(title, abstract) for title, abstract in zip(each['c_title'], each['c_abstract'])]
                disci_one, disci_two = '\n'.join(disci_one), '\n'.join(disci_two)
                # budget system
                if budget:
                    remaining_budget = budget_num - sum(responses)
                    if remaining_budget == 0:
                        break
                    input_prompt = prompt_exp_2_budget % (remaining_budget, disci_one, disci_two)
                else:
                    input_prompt = prompt_exp_2 % (disci_one, disci_two)

                # change prompt here for each task
                response_txt, logprobs = self.client.generate(sys_prompt=sys_prompt_critical if critical else sys_prompt, input_prompt=input_prompt)                
                # This regex uses named capture groups for verdict and reason.
                pattern = r"Your verdict:\s*(?P<verdict>Yes|No).?\s*Your reason:\s*(?P<reason>.+).?\s*Confidence score:\s*(?P<score>\d+).?"
                match = re.search(pattern, response_txt, re.IGNORECASE)
                verdict, reason, verb_score = None, None, None
                if match:
                    data = match.groupdict()
                    # Map verdict values to boolean
                    verdict_mapping = {"yes": True, "no": False}
                    # Convert verdict to lowercase to match our mapping keys
                    data["verdict"] = verdict_mapping.get(data["verdict"].lower())
                    # Optionally, strip any extra spaces from the reason text
                    data["reason"] = data["reason"].strip()
                    verdict, reason, verb_score = data["verdict"], data["reason"], int(data["score"])
                else:
                    if verbose:
                        print(f"Match not found. Response: {response_txt}")
                    continue
                # update the result collection
                ids.append(each['id'])
                responses.append(verdict)
                labels.append(each['y_true'])
                reasons_pred.append(reason)
                verb_conf.append(verb_score)
                response_logprobs.append(logprobs)
                if i % checkpoint_len == 0:
                    if verbose:
                        print(f"Making checkpoint on file: {file_name}...")
                    # collect result and put into the df
                    data = {
                        "id": ids,
                        "y_true": labels,
                        "y_pred": responses,
                        "reasons": reasons_pred,
                        "verb_conf": verb_conf,
                        "log_probs": response_logprobs
                    }
                    # save to json file
                    pd.DataFrame(data).to_json('/'.join([self.save_path, out_file_name]), indent=2, index=False, orient='records')
            # collect result and put into the df
            data = {
                "id": ids,
                "y_true": labels,
                "y_pred": responses,
                "reasons": reasons_pred,
                "verb_conf": verb_conf,
                "log_probs": response_logprobs
            }
            # save to json file
            pd.DataFrame(data).to_json(os.path.join(self.save_path, out_file_name), indent=2, index=False, orient='records')

    # exp_3
    def recommend_task(self, file_names: list = [f"data_exp_3_{i+1}.json" for i in range(2)], verbose: bool = False, checkpoint_len: int = 1):
        # check if we want a critical llm
        critical = self.kwargs.get("critical", False)
        if verbose:
            print(f"Critical llm: {critical}")
        for idx, file_name in enumerate(file_names):
            out_file_name = f"exp_3_{idx+1}_{self.model_name}"
            # assemble out file name
            if critical:
                out_file_name += "_critical"
            out_file_name += ".json"
            # checkpoint system to obtain start point
            cached_idx = 0
            cached = False
            # check if the file exists and load it
            if os.path.exists(os.path.join(self.save_path, out_file_name)):
                df_cached = pd.read_json(os.path.join(self.save_path, out_file_name))
                cached_idx = df_cached.shape[0]
                cached = True
            if verbose:
                print(f"Already processed {cached_idx} samples from {out_file_name}. Continuing from there...")
                print(f"Doing data file: {file_name}...")
            # load benchmark data
            df = pd.read_json(os.path.join(self.data_path, file_name))#[:5] # first try 5 samples to ensure works well
            df_size = df.shape[0] - cached_idx
            # break the function if no data
            if df_size == 0:
                if verbose:
                    print(f"No new data to process in {file_name}.")
                # return
            # slice the dataframe to start from the last processed index
            df = df[cached_idx:]
            ids, responses, labels, verb_conf, response_logprobs = [], [], [], [], []
            if cached:
                # load the cached data
                ids = df_cached['id'].tolist()
                responses = df_cached['y_pred'].tolist()
                labels = df_cached['list'].tolist()
                verb_conf = df_cached['verb_conf'].tolist()
                response_logprobs = df_cached['log_probs'].tolist()
            # main loop
            i = 0
            for i, each in tqdm(df.iterrows(), total=df_size):
                """
                Swiss tournament using LLM as judge. No budget here as this is ranking task.
                """
                # the given starting paper A
                context = each[["start_title", "start_abstract"]].to_dict()
                # each["list"]: {"title": [], "abstract": []}
                Papers = [Player(title=title, abstract=abstract) for title, abstract in zip(each["list"]["title"], each["list"]["abstract"])]
                # test_paper = Papers[:3] # small amount of test paper
                paper_rank, logprob, verb_score = swiss_tournament(Papers, context, critical, self.client, 10, verbose=verbose)
                # update the result collection
                ids.append(each['id'])
                responses.append([{"title": paper.title, "abstract": paper.abstract, "score": paper.score} for paper in paper_rank])
                labels.append(each['list'])
                verb_conf.append(verb_score)
                response_logprobs.append(logprob)
                if i % checkpoint_len == 0:
                    if verbose:
                        print(f"Making checkpoint on file: {file_name}...")
                    # collect result and put into the df
                    data = {
                        "id": ids,
                        "list": labels,
                        "y_pred": responses,
                        "verb_conf": verb_conf,
                        "log_probs": response_logprobs,
                    }
                    # save to json file
                    pd.DataFrame(data).to_json(os.path.join(self.save_path, out_file_name), indent=2, index=False, orient='records')
            if verbose:
                print(f"Total Comparisons made (i={i}): {len(verb_conf)}")
            # collect result and put into the df
            data = {
                "id": ids,
                "list": labels,
                "y_pred": responses,
                "verb_conf": verb_conf,
                "log_probs": response_logprobs,
            }
            # save to json file
            pd.DataFrame(data).to_json(os.path.join(self.save_path, out_file_name), indent=2, index=False, orient='records')
    
    def recommend_ranking_task(self, file_names: list = [f"data_exp_3_{i+1}.json" for i in range(2)], number_of_papers=10, verbose: bool = False, checkpoint_len: int = 1):
        # check if we want a critical llm
        critical = self.kwargs.get("critical", False)
        if verbose:
            print(f"Critical llm: {critical}")
        for idx, file_name in enumerate(file_names):
            out_file_name = f"exp_3r_{idx+1}_{self.model_name}"
            # assemble out file name
            if critical:
                out_file_name += "_critical"
            out_file_name += ".json"
            # checkpoint system to obtain start point
            cached_idx = 0
            cached = False
            # check if the file exists and load it
            if os.path.exists(os.path.join(self.save_path, out_file_name)):
                df_cached = pd.read_json(os.path.join(self.save_path, out_file_name))
                cached_idx = df_cached.shape[0]
                cached = True
            if verbose:
                print(f"Already processed {cached_idx} samples. Continuing from there...")
                print(f"Doing data file: {file_name}...")
            # load benchmark data
            df = pd.read_json(os.path.join(self.data_path, file_name))#[:5] # first try 5 samples to ensure works well
            df_size = df.shape[0] - cached_idx
            # break the function if no data
            if df_size == 0:
                if verbose:
                    print(f"No new data to process in {file_name}.")
                return
            # slice the dataframe to start from the last processed index
            df = df[cached_idx:]
            ids, y_true, responses, reasons, verb_conf, response_logprobs, list_strings = [], [], [], [], [], [], []
            if cached:
                # load the cached data
                ids = df_cached['id'].tolist()
                y_true = df_cached['y_true'].tolist()
                responses = df_cached['y_pred'].tolist()
                reasons = df_cached['reason'].tolist()
                verb_conf = df_cached['verb_conf'].tolist()
                response_logprobs = df_cached['log_probs'].tolist()
                list_strings = df_cached['list_strings'].tolist()
            # main loop
            for i, each in tqdm(df.iterrows(), total=df_size):
                context = each[["start_title", "start_abstract"]].to_dict()
                title = context["start_title"]
                abstract = context["start_abstract"]
    
                target = each["target_paper"]
                target_zip = list(zip(target["title"], target["abstract"]))
                
                list_of_papers = each["list"]
                list_zip = list(zip(list_of_papers["title"], list_of_papers["abstract"]))
                
                list_complete = list_zip + target_zip
                random.shuffle(list_complete)
                true_numbers = [str(list_complete.index(x)+1) for x in target_zip]
                
                string = ";\n".join([re.sub(r'\s+', ' ', f"{n+1}) Title: {p[0]}; Abstract: {p[1]}").strip() for n, p in enumerate(list_complete)])
                
                input_prompt = prompt_exp_3_ranking % (number_of_papers, title, abstract, list_of_papers)
    
                response_txt, logprobs = self.client.generate(sys_prompt=sys_prompt_critical if critical else sys_prompt, input_prompt=input_prompt)

                verdict, reason, verb_score = None, None, None
                try:
                    data = json.loads(response_txt)
                    
                    verdict = [str(item) for item in data["list"]]
                    reason = data["reasoning"]
                    verb_score = int(data["confidence_score"])
                except:
                    if verbose:
                        print(f"Match not found. Response: {response_txt}")
                    continue
                
                
                ids.append((each['id'], each['start_id']))
                y_true.append(true_numbers)
                responses.append(verdict)
                reasons.append(reason)
                verb_conf.append(verb_score)
                response_logprobs.append(logprobs)
                list_strings.append(string)
                
                if i % checkpoint_len == 0:
                    if verbose:
                        print(f"Making checkpoint on file: {file_name}...")
                    # collect result and put into the df
                    data = {
                        "id": ids,
                        "y_true": y_true,
                        "y_pred": responses,
                        "reason": reasons,
                        "verb_conf": verb_conf,
                        "log_probs": response_logprobs,
                        "list_string": list_strings,
                    }
                    # save to json file
                    pd.DataFrame(data).to_json(os.path.join(self.save_path, out_file_name), indent=2, index=False, orient='records')
            
            # collect result and put into the df
            data = {
                "id": ids,
                "y_true": y_true,
                "y_pred": responses,
                "reason": reasons,
                "verb_conf": verb_conf,
                "log_probs": response_logprobs,
                "list_string": list_strings,
            }
            # save to json file
            pd.DataFrame(data).to_json(os.path.join(self.save_path, out_file_name), indent=2, index=False, orient='records')
    

# Test code to ensure everything is right:
if __name__ == "__main__":
    config_path = './config.yml'
    task_config = load_config(config_path)['task_config']
    handler = TaskHandler(provider='gemini', model_name="gemini-2.0-flash", lm_config_path="./config.yml", **task_config)
    # handler = TaskHandler(provider='gpt', model_name="gpt-4o-mini-2024-07-18", lm_config_path="./config.yml", **task_config)
    # handler = TaskHandler(provider='deepseek', model_name="deepseek-reasoner", lm_config_path="./config.yml", **task_config)
    # handler = TaskHandler(provider='llama', model_name="llama-3.3-70b-instruct", lm_config_path="./config.yml", **task_config)

    task_func = handler["identification"]
    task_func(verbose=True)

