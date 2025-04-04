# from transformers import AutoTokenizer, AutoModel
import random
import json
import yaml


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def load_config(config_path='config.yml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

sys_prompt = "You are a researcher in building interdisciplinary research projects. Use your your existing knowledge over several distinct areas to provide constructive verdicts on the feasibility of building interdisciplinary research projects."

prompt_exp_1 = """
Read the title and abstract of a given academic paper and identify whether this is an interdisciplinary research paper. 
The official definition of a typical interdisciplinary paper can be found below: 
“Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
Think carefully to make your decision, answer "Yes" when this is a valid IDR paper. Otherwise, answer "No".
-----
Paper title: %s;
Paper abstract: %s;
-----
Use the template (in this format, with no markdown and lines separated by '\n') below to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
"""

prompt_exp_2 = """
Read the abstract of the two academic papers that introduces ideas from Interdisciplinary Research disciplines and decide whether you can extract one or more concepts from both sides to create a novel multidisciplinary research idea. 
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to make your decision, and you should only answer "Yes" when this multidisciplinary idea meets ALL of the standards above. Otherwise, you should answer "No".
-----
Paper 1 title: %s;
Paper 1 abstract: %s; 
-----
Paper 2 title: %s;
Paper 2 abstract: %s;
-----
Use the template (in this format, with no markdown and lines separated by '\n') to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Your reason: {A short paragraph less than 50 words briefly describes your reasons that you made the verdict above.}
"""

prompt_exp_3 = """
In this task, you are given a main paper introducing the key concepts that provides certain parts in a Interdisciplinary idea as well as two candidate papers that forms the remaining parts of a Interdisciplinary idea. Compare them and select which one is better to pair with the main paper in forming a multidisciplinary idea. 
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
-----
Main paper title: %s;
Main paper abstract: %s;
-----
Paper 1 title: %s;
Paper 1 abstract: %s; 
-----
Paper 2 title: %s;
Paper 2 abstract: %s;
-----
Use the template (in this format, with no markdown and lines separated by '\n') to provide your answer.
Your choice: {A simple answer containing either "Paper 1" or "Paper 2".}
"""


# Test code to ensure everything is right:
if __name__ == "__main__":
    pass



