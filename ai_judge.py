import yaml
from pandas import Series
from jinja2 import Environment, meta
from openai import OpenAI
import streamlit as st
import logging.config
import json


with open('judge_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.config.dictConfig(config['logging'])
logger = logging.getLogger('ai_judge')


def replace_placeholders(row: Series, prompt_template: str) -> str:
    env = Environment()

    return env.from_string(prompt_template).render(row=row)


class AiJudge:
    def __init__(self):
        self.system_message = config['open_ai']['system_message']
        self.model_config = config['open_ai']['model_config']

    def call_llm(self, prompt: str,  row: Series) -> dict:
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(messages=messages, **self.model_config)

        content_json = response.choices[0].message.content
        logger.debug(f"LLM Response: {content_json}")
        return json.loads(content_json)

