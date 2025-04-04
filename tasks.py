# import random
import re
import numpy as np
from tqdm import tqdm
from lm_client import LM_Client
from utils import *
# from evaluation_v1 import calculate_mrr, ndcg_at_10
import pandas as pd
from typing import *
from copy import deepcopy


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
        }
        self.client = LM_Client(provider=self.provider, model_name=self.model_name, config_path=self.lm_config_path)
        self.kwargs = kwargs

    def __getitem__(self, key: Literal["classification", "identification", "recommendation"]):
        """Enables retrieval using Object[key]."""
        return self.task_mapping.get(key, "No matching tasks identified")
    # exp_1
    def identify_task(self, file_name: str = "data_exp_1.csv", verbose: bool = False):
        df = pd.read_csv('/'.join([self.data_path, file_name]))[:5] # first try 5 samples to ensure works well
        df["log_probs"] = np.nan
        responses, response_logprobs = [], []
        # check if need budget:
        budget = self.kwargs.get("budget_mode", None)
        budget_num = self.kwargs.get("budget_num", None)
        if verbose:
            print(f"Doing data file: {file_name}...")
            print(f"Budget mode: {budget}, num of yes to say: {budget_num}")

        for _, each in tqdm(df.iterrows(), total=len(df)):
            title, abstract = each['title'], each['abstract']
            # budget system
            if budget:
                remaining_budget = budget_num - sum(responses)
                if remaining_budget == 0:
                    # pad the remaining sample negative and exit loop
                    responses += [0] * (len(df) - len(responses))
                    break
                input_prompt = prompt_exp_1_budget % (remaining_budget, title, abstract)
            else:
                input_prompt = prompt_exp_1 % (title, abstract)
            # change prompt here for each task
            response_txt, logprobs = self.client.generate(sys_prompt=sys_prompt, input_prompt=input_prompt)
            # get logprobs if included in the lm config
            if logprobs: 
                response_logprobs.append(logprobs)

            # This regex uses named capture groups for verdict and reason.
            verdict = None
            pattern = r"Your verdict:\s*(?P<verdict>Yes|No)"
            match = re.search(pattern, response_txt, re.IGNORECASE)
            if match:
                # Extract the verdict value
                verdict_value = match.group("verdict")
                # Map verdict values to boolean
                verdict_mapping = {"yes": True, "no": False}
                result = {"verdict": verdict_mapping.get(verdict_value.lower())}
                verdict = result["verdict"]
            # update the result collection
            responses.append(verdict)
        # collect result and put into the df
        if response_logprobs:
            # Don't forget to pad logprobs as well
            if len(response_logprobs) < len(df):
                response_logprobs += [None]*(len(df) - len(response_logprobs))
            df["log_probs"] = response_logprobs
        df["y_pred"] = responses
        # save to json file
        out_file_name = f"exp_1_{self.model_name}.json" if not budget else f"exp_1_budget_{self.model_name}.json"
        df.to_json('/'.join([self.save_path, out_file_name]), indent=2, index=False, orient='records')
    # exp_2
    def classify_task(self, file_names: list = [f"data_exp_2_{i+1}.csv" for i in range(3)], verbose: bool = False): 
        # check if need budget:
        budget = self.kwargs.get("budget_mode", None)
        budget_num = self.kwargs.get("budget_num", None)
        if verbose:
            print(f"Budget mode: {budget}, num of yes to say: {budget_num}")
        for idx, file_name in enumerate(file_names):
            df = pd.read_csv('/'.join([self.data_path, file_name]))[:5] # first try 5 samples to ensure works well
            df["log_probs"] = np.nan
            responses, reasons, response_logprobs = [], [], []
            if verbose:
                print(f"Doing data file: {file_name}...")
            for _, each in tqdm(df.iterrows(), total=len(df)):
                title_1, title_2, abstract_1, abstract_2 = each['b_title'], each['c_title'], each['b_abstract'], each['c_abstract']
                # budget system
                if budget:
                    remaining_budget = budget_num - sum(responses)
                    if remaining_budget == 0:
                        # pad the remaining sample negative and exit loop
                        responses += [0] * (len(df) - len(responses))
                        break
                    input_prompt = prompt_exp_2_budget % (remaining_budget, title_1, abstract_1, title_2, abstract_2)
                else:
                    input_prompt = prompt_exp_2 % (title_1, abstract_1, title_2, abstract_2)

                # change prompt here for each task
                response_txt, logprobs = self.client.generate(sys_prompt=sys_prompt, input_prompt=input_prompt)
                # get logprobs if included in the lm config
                if logprobs: 
                    response_logprobs.append(logprobs)
                # This regex uses named capture groups for verdict and reason.
                pattern = r"Your verdict:\s*(?P<verdict>Yes|No)\s*\n*Your reason:\s*(?P<reason>.+)"
                match = re.search(pattern, response_txt, re.IGNORECASE)
                verdict, reason = None, None
                if match:
                    data = match.groupdict()
                    # Map verdict values to boolean
                    verdict_mapping = {"yes": True, "no": False}
                    # Convert verdict to lowercase to match our mapping keys
                    data["verdict"] = verdict_mapping.get(data["verdict"].lower())
                    # Optionally, strip any extra spaces from the reason text
                    data["reason"] = data["reason"].strip()
                    verdict, reason = data["verdict"], data["reason"]
                # update the result collection
                responses.append(verdict)
                reasons.append(reason)
            """
            collect result and put into the df
            """
            if response_logprobs:
                # Don't forget to pad logprobs as well
                if len(response_logprobs) < len(df):
                    response_logprobs += [None]*(len(df) - len(response_logprobs))
                df["log_probs"] = response_logprobs
            df["y_pred"] = responses
            # finally pad reasons
            if len(reasons) < len(df):
                reasons += [None]*(len(df) - len(reasons))
            df["reasons"] = reasons
            # save to json file
            out_file_name = f"exp_2_{idx+1}_{self.model_name}.json" if not budget else f"exp_2_{idx+1}_budget_{self.model_name}.json"
            df.to_json('/'.join([self.save_path, out_file_name]), indent=2, index=False, orient='records')
    
    # exp_3
    def recommend_task(self, file_names: list = [f"data_exp_3_{i+1}.json" for i in range(2)], verbose: bool = False):
        for idx, file_name in enumerate(file_names):
            df = pd.read_json('/'.join([self.data_path, file_name]))[:2] # first try 5 samples to ensure works well
            df["log_probs"] = np.nan
            responses, response_logprobs = [], []
            if verbose:
                print(f"Doing data file: {file_name}...")
            for _, each in tqdm(df.iterrows(), total=len(df)):
                """
                Insertion sort using LLM as judge. No budget here as this is ranking task.
                """
                ranked_lst = [(each["list"]["title"][0], each["list"]["abstract"][0])]
                # the given starting paper A
                start_title, start_abstract = each["start_title"], each["start_abstract"]
                # collect answer logprob (avg.) for each comparison 
                logprobs_mid = []
                lst2rank_length = len(each["list"]["title"][:10]) # first try 10 samples to rank
                assert lst2rank_length >= 2
                if verbose:
                    print(f"Length of a list to rank for recommendation: {lst2rank_length}")
                for title, abstract in zip(each["list"]["title"][1:10], each["list"]["abstract"][1:10]): # first try 10 samples to rank
                    inserted = False
                    for i, sorted in enumerate(ranked_lst):
                        # change prompt here for each task
                        input_prompt = prompt_exp_3 % (start_title, start_abstract, sorted[0], sorted[1], title, abstract)
                        response_txt, logprobs = self.client.generate(sys_prompt=sys_prompt, input_prompt=input_prompt)
                        logprobs = np.average(logprobs['logprob'])
                        logprobs_mid.append(logprobs)
                        verdict = False
                        # This regex uses named capture groups for verdict and reason.
                        pattern = r"Your choice:\s*(?P<verdict>Paper 1|Papepr 2)"
                        match = re.search(pattern, response_txt, re.IGNORECASE)
                        if match:
                            # Extract the verdict value
                            verdict_value = match.group("verdict")
                            # Map verdict values to boolean
                            verdict_mapping = {"Paper 2": True, "Paper 1": False}
                            verdict = verdict_mapping.get(verdict_value.lower())
                        
                        if verdict:
                            ranked_lst.insert(i, (title, abstract))
                            inserted = True
                            break
                    if not inserted:
                        ranked_lst.insert(i, (title, abstract))
                # update the result collection
                response_logprobs.append(logprobs_mid)
                responses.append(ranked_lst)
            """
            collect result and put into the df
            """
            if response_logprobs:
                df["log_probs"] = response_logprobs
            df["y_pred"] = responses
            # save to json file
            out_file_name = f"exp_3_{idx+1}_{self.model_name}.json"
            df.to_json('/'.join([self.save_path, out_file_name]), indent=2, index=False, orient='records')

# Test code to ensure everything is right:
if __name__ == "__main__":
    config_path = './config.yml'
    task_config = load_config(config_path)['task_config']
    handler = TaskHandler(provider='gpt', model_name="gpt-4o-mini-2024-07-18", lm_config_path="./config.yml", **task_config)
    task_func = handler["classification"]
    task_func(verbose=True)

