import openai
import os
from fireworks.client import Fireworks
import yaml

with open('config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except:
        print("Invalid config.yml file.")

fw_key = config["llm_config"]["fw_key"]
openai.api_key = config["llm_config"]["oai_key"]



def getGPT4oMiniResponse(prompt, system_prompt, model="gpt-4o-mini", max_tokens=500):
    response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{prompt}"}
            ],
            temperature=0,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0
        ).choices[0]

    return response.message.content


def getO1MiniResponse(prompt, system_prompt, model="o1-mini"):
    response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ],
        ).choices[0]

    return response.message.content


def getO3MiniResponse(prompt, system_prompt, model="o3-mini"):
    response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ],
        ).choices[0]

    return response.message.content


def getLlama3p1Response(prompt, system_prompt):
    client = Fireworks(api_key=fw_key)
    response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{prompt}"}
            ],
        ).choices[0]
    
    return response.message.content



llmResponseGetters = {
    "4o-mini": getGPT4oMiniResponse,
    "o1-mini": getO1MiniResponse,
    "o3-mini": getO3MiniResponse,
    "llama-3p1": getLlama3p1Response,
}