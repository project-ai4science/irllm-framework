from openai import OpenAI
from fireworks.client import Fireworks
# from google import genai
import os
import yaml
from typing import *
from utils import load_config
    
CONFIG = load_config()["llm_client_config"]
CREDENTIALS = CONFIG["credentials"]
MODELS = CONFIG["model_list"]

class LM_Client():

    def __init__(self, provider: str, model_name: str, config_path: str = None, **kwargs):
        # make sure the models are in the exp list
        assert provider in MODELS
        assert model_name in MODELS[provider]
        self.clients = self._init_clients()
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
            # Create a new dictionary with keys renamed according to the mapping
            self.model_config = {
                # "max_output_tokens": self.model_config.get("max_tokens", 500),
                "temperature": self.model_config.get("temperature", 1.0),
            }

        # Merge any additional keyword arguments into api_params.
        self.model_config.update(kwargs)

    def _init_clients(self):
        clients = {}
        if "openai_key" in CREDENTIALS:
            clients["openai"] = OpenAI(api_key=CREDENTIALS["openai_key"])
        if "firework_key" in CREDENTIALS:
            clients["firework"] = Fireworks(api_key=CREDENTIALS["firework_key"])
        if "openrouter_key" in CREDENTIALS:
            clients["openrouter"] = OpenAI(
                api_key=CREDENTIALS["openrouter_key"],
                base_url="https://openrouter.ai/api/v1"
            )
        if "deepseek_key" in CREDENTIALS:
            clients["deepseek"] = OpenAI(
                api_key=CREDENTIALS["deepseek_key"],
                base_url="https://api.deepseek.com"
            )
        if "grok_key" in CREDENTIALS:
            clients["grok"] = OpenAI(
                api_key=CREDENTIALS["grok_key"],
                base_url="https://api.x.ai/v1"
            )
        # if "gemini_key" in CREDENTIALS:
        #     genai.configure(api_key=CREDENTIALS["gemini_key"])

        return clients

    def _load_config(self, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            all_configs = yaml.safe_load(f)
            lm_configs = all_configs['llm_client_config']
            return lm_configs

    def generate(self, input_prompt: str, sys_prompt: str = None):
        input_msg = [
            {"role": "user" if 'o1' in self.model_name else "system", "content": f"{sys_prompt}" if sys_prompt else "You are an helpful assitant."},
            {"role": "user", "content": f"{input_prompt}"}
        ]
        logprobs = None

        if self.provider == 'gpt':
            if ('o3' in self.model_name):
                response = self.clients['openai'].responses.create(
                    model = self.model_name,
                    instructions = sys_prompt if sys_prompt else "You are an helpful assitant.",
                    input = input_prompt,
                    **self.model_config
                )
                response_txt = response.output_text
                print(response_txt)

            else:
                response = self.clients['openai'].chat.completions.create(
                    model = self.model_name,
                    messages = input_msg,
                    **self.model_config
                ).choices[0]
                response_txt = response.message.content

                if response.logprobs:
                    logprobs = response.logprobs.content
                    logprobs = {"token": [each.token for each in logprobs], "logprob": [each.logprob for each in logprobs]}

        elif self.provider == 'llama':
            response = self.clients['firework'].chat.completions.create(
                model = f"accounts/fireworks/models/{self.model_name}",
                messages = input_msg,
                **self.model_config
            ).choices[0]
            response_txt = response.message.content

            if self.model_config["logprobs"] is not None:
                logprobs = response.logprobs

        elif self.provider == 'openrouter':
            response = self.clients['openrouter'].chat.completions.create(
                model = self.model_name,
                messages = input_msg,
                **self.model_config
            ).choices[0]
            response_txt = response.message.content

        elif self.provider == 'deepseek':
            response = self.clients['deepseek'].chat.completions.create(
                model = self.model_name,
                messages = input_msg,
                **self.model_config
            ).choices[0]
            response_txt = response.message.content

        elif self.provider == 'grok':
            response = self.clients['grok'].chat.completions.create(
                model = self.model_name,
                messages = input_msg,
                **self.model_config
            ).choices[0]
            response_txt = response.message.content

        # elif self.provider == 'gemini':
        #     model = genai.GenerativeModel(self.model_name)
        #     generation_config = genai.types.GenerationConfig(
        #         system_instruction=sys_prompt,
        #         model_config=genai.types.ModelConfig(**self.model_config)
        #     )
        #     response = model.generate_content(
        #         input_prompt,
        #         generation_config=generation_config
        #     )
        #     response_txt = response.text

        else:
            print("Not yet implemented!")
            response_txt = None

        return response_txt, logprobs











