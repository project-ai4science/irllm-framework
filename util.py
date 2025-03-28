# from transformers import AutoTokenizer, AutoModel
import random
import json
from llm import llmResponseGetters
import yaml

with open('config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except:
        print("Invalid config.yml file.")


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to("cpu")


# def scibertEncode(string):
#     inputs = tokenizer(
#         string,
#         padding=True,
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     ).to("cpu")

#     outputs = model(**inputs)

#     return outputs[1]


def identify(title, abstract):
    llm_answer = identifyIDRPaper(title, abstract)
    
    verdict = llm_answer.split("Your verdict: ")[-1].replace("\n", " ").split(".")[0].strip().lower()
    verdict = True if verdict == "yes" else False if verdict == "no" else "None" if verdict is None else verdict
    
    return verdict

def identifyIDRPaper(title, abstract):
    system = "You are a researcher in building interdisciplinary research projects. Use your your existing knowledge over several distinct areas to provide constructive verdicts on the feasibility of building interdisciplinary research projects."
    prompt = (
"""
Read the title and abstract of a given academic paper and identify whether this is an interdisciplinary research paper. 
The official definition of a typical interdisciplinary paper can be found below: 
“Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
Think carefully to make your decision, answer "Yes" when this is a valid IDR paper. Otherwise, answer "No".
"""
f"""
Paper title: {title};  Paper abstract: {abstract};
"""
"""
Use the template (in this format, with no markdown and lines separated by \n) below to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
"""
)
    
    return llmResponseGetters[config["llm_config"]["llm"]](prompt, system)


def classify(paper_b, paper_c):
    llm_answer = classifyIDRPaper(paper_b, paper_c)
    
    verdict = llm_answer.split("Your verdict: ")[-1].split("\n")[0].split(".")[0].strip().lower()
    verdict = True if verdict == "yes" else False if verdict == "no" else verdict
    if verdict is None:
        print(llm_answer)
    
    reason = llm_answer.split("Your reason: ")[-1].strip()
    
    return verdict, reason



def classifyIDRPaper(paperB, paperC):
    system = "You are a researcher in building new multidisciplinary research projects. Use your your existing knowledge over several distinct areas to provide constructive verdicts on the feasibility of building multidisciplinary research projects."
    prompt = (
"""
Read the abstract of the two academic papers that introduces ideas from Interdisciplinary Research disciplines and decide whether you can extract one or more concepts from both sides to create a novel multidisciplinary research idea. 
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to make your decision, and you should only answer "Yes" when this multidisciplinary idea meets ALL of the standards above. Otherwise, you should answer "No".

"""
f"""
Paper 1: {paperB}; 

Paper 2: {paperC};
"""
"""
Use the template (in this format, with no markdown and lines separated by \n) to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Your reason: {A short paragraph less than 50 words briefly describes your reasons that you made the verdict above.}
"""
)
    
    return llmResponseGetters[config["llm_config"]["llm"]](prompt, system)


def classify(paper_b, paper_c):
    llm_answer = classifyIDRPaper(paper_b, paper_c)
    
    verdict = llm_answer.split("Your verdict: ")[-1].split("\n")[0].split(".")[0].strip().lower()
    verdict = True if verdict == "yes" else False if verdict == "no" else verdict
    if verdict is None:
        print(llm_answer)
    
    reason = llm_answer.split("Your reason: ")[-1].strip()
    
    return verdict, reason



def recommendIDRPaper(paperB, papers_string, n=5):
    system = "You are a researcher interested in build new multidisciplinar researches. Considering your existing knowledge over several distinct areas, you can act as a supervisor and suggest new paper in order to propose a new multidisciplinary work."
    prompt = (
f"""
In this task, you are given a main paper introducing the key concepts that provides certain parts in a multidisciplinary idea as well as a list of candidate papers that forms the remaining parts of a multidisciplinary idea. Compare them and decide which one of the candidates matches with the main paper better in forming this multidisciplinary idea. 
The list of papers should consist of the letters associated with them and with exactly {n} papers (no more, no less). The list should be ordered (best papers first) and first paper of the list must be the best choice to form a multidisciplinary idea.
Keep in mind a good multidisciplinary research idea includes the following standards: 
* This research idea should be multidisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem
"""
f"""
Main paper: {paperB}; 

List of candidate papers:
{papers_string}
"""
"""
Follow the format in the example to provide your response (only the JSON in one single line with no line breaks and with no markdown): {"reasoning": "reasoning_string", "list": ["List", "of", "letters"]}
"""
)
    
    return llmResponseGetters[config["llm_config"]["llm"]](prompt, system)


def recommend(paper_main, paper_2, false_ids):
    false_papers = [data.loc[data.id == fid].b_concat.values[0] for fid in false_ids]
    false_papers += [data.loc[data.id == fid].c_concat.values[0] for fid in false_ids]
    random.shuffle(false_papers)
    false_papers = false_papers[:9]
    
    papers = [paper_2, *false_papers]
    random.shuffle(papers)
    
    true_idx = papers.index(paper_2)
    relevant = LETTERS[true_idx]
    
    papers_zip = zip(papers, LETTERS[:len(papers)])
    papers_string = "; ".join([f'{letter}) {paper}' for paper, letter in papers_zip])

    llm_answer = recommendIDRPaper(paper_main, papers_string).replace("\\", "\\\\").replace("\\\\\"", "\\\"").replace("\\\\n", " ")

    r_json = json.loads(llm_answer)

    reasoning = r_json["reasoning"]
    rec_list = r_json["list"]
    
    return papers_string, relevant, rec_list, reasoning