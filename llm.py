import openai
import os
from fireworks.client import Fireworks
import yaml
import google.generativeai as genai

with open('config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except:
        print("Invalid config.yml file.")

fw_key = config["llm_config"]["fw_key"]
openai.api_key = config["llm_config"]["oai_key"]
gemini_key = config["llm_config"]["gemini_key"]
deepseek_key = config["llm_config"]["deepseek_key"]
xai_key = config["llm_config"]["xai_key"]


genai.configure(api_key=gemini_key)

def getGPT4oMiniResponse(prompt, system_prompt, model="gpt-4o-mini", max_tokens=500):
    response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{prompt}"}
            ],
            temperature=0.8,
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


def getDeepSpeekR1Response(prompt, system_prompt, model='deepseek-reasoner'):
     # see models options at https://api-docs.deepseek.com/quick_start/pricing
    response = openai.ChatCompletion.create(
            model=model,
            openai_api_base='https://api.deepseek.com',
            openai_api_key=deepseek_key,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{prompt}"}
            ],
        ).choices[0]

    return response.message.content


def getGrok2Response(prompt, system_prompt, model='grok-2-latest'):
    # see models options at https://docs.x.ai/docs/models?cluster=us-east-1#models-and-pricing
    response = openai.ChatCompletion.create(
            model=model,
            openai_api_base='https://api.x.ai/v1',
            openai_api_key=xai_key,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{prompt}"}
            ],
        ).choices[0]

    return response.message.content


def getGemini2FlashResponse(prompt, system_prompt, model="gemini-2.0-flash-001"):
    # see models options at https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#supported-models
    model_gen = genai.GenerativeModel(model)
    chat = model_gen.start_chat(history=[])
    response = chat.send_message(f"{system_prompt}\n{prompt}")
    return response.text



llmResponseGetters = {
    "4o-mini": getGPT4oMiniResponse,
    "o1-mini": getO1MiniResponse,
    "o3-mini": getO3MiniResponse,
    "llama-3p1": getLlama3p1Response,
    "deepseek-reasoner": getDeepSpeekR1Response,
    "grok-2-latest": getGrok2Response,
    "gemini-2.0-flash-001": getGemini2FlashResponse,
}