from openai import OpenAI
from fireworks.client import Fireworks
# import google.generativeai as genai
import os
import yaml
from typing import *
from utils import load_config
    
CONFIG = load_config()["llm_client_config"]
CREDENTIALS = CONFIG["credentials"]
MODELS = CONFIG["model_list"]

openai_client = OpenAI(api_key=CREDENTIALS["openai_key"])
firewok_client = Fireworks(api_key=CREDENTIALS["firework_key"])


class LM_Client():

    def __init__(self, provider: str, model_name: str, config_path: str = None, **kwargs):
        # make sure the models are in the exp list
        assert provider in MODELS
        assert model_name in MODELS[provider]
        self.provider = provider
        self.model_name = model_name
        self.lm_config = self._load_config(config_path) if config_path else dict()

        # gpt-4 series uses chat.completions
        self.model_config = {
            "max_tokens": self.lm_config.get("max_tokens", 500),
            "temperature": self.lm_config.get("temperature", 1.0),
            "logprobs": self.lm_config.get("logprobs", None),
            "frequency_penalty": self.lm_config.get("frequency_penalty", 0),
            "presence_penalty": self.lm_config.get("presence_penalty", 0)
        }
        # gpt-o series uses chat.responses
        if ('o1' in self.model_name) or ('o3' in self.model_name):
            # # Define a mapping from old keys to new keys
            # rename_map = {
            #     'max_tokens': 'max_output_tokens',
            #     # 'old_key2': 'new_key2'
            # }

            # Create a new dictionary with keys renamed according to the mapping
            self.model_config = {
                # "max_output_tokens": self.model_config.get("max_tokens", 500),
                "temperature": self.model_config.get("temperature", 1.0),
            }

        # Merge any additional keyword arguments into api_params.
        self.model_config.update(kwargs)


    def _load_config(self, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
            lm_configs = all_configs['llm_client_config']
            return lm_configs

    def generate(self, input_prompt: str, sys_prompt: str = None):
        input_msg = [
            {"role": "system" , "content": f"{sys_prompt}" if sys_prompt else "You are an helpful assitant."},
            {"role": "user", "content": f"{input_prompt}"}
        ]
        logprobs = None

        if self.provider == 'gpt':
            if ('o1' in self.model_name) or ('o3' in self.model_name):
                response = openai_client.responses.create(
                    model = self.model_name,
                    instructions = sys_prompt if sys_prompt else "You are an helpful assitant.",
                    input = input_prompt,
                    **self.model_config
                )
                response_txt = response.output_text
                print(response_txt)

            else:
                response = openai_client.chat.completions.create(
                    model = self.model_name,
                    messages = input_msg,
                    **self.model_config
                ).choices[0]
                response_txt = response.message.content

                if response.logprobs:
                    logprobs = response.logprobs.content
                    logprobs = {"token": [each.token for each in logprobs], "logprob": [each.logprob for each in logprobs]}

        elif self.provider == 'llama':
            response = firewok_client.chat.completions.create(
                model = f"accounts/fireworks/models/{self.model_name}",
                messages = input_msg,
                **self.model_config
            ).choices[0]
            response_txt = response.message.content

            if self.model_config["logprobs"] is not None:
                logprobs = response.logprobs

        elif self.provider == 'deepseek':
            pass

        elif self.provider == 'grok':
            pass

        elif self.provider == 'gemini':
            pass

        else:
            print("Not yet implemented!")
            response_txt = None

        return response_txt, logprobs











