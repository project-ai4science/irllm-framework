# from transformers import AutoTokenizer, AutoModel
# import random
import numpy as np
import re
# import json
import yaml
from typing import *
# from lm_client import LM_Client


# LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def load_config(config_path='config.yml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

class Player:
    def __init__(self, title, abstract):
        self.title = title
        self.abstract = abstract
        self.score = 0
        self.opponents = []  # Keep track of player names this player has already faced
        self.bye = False     # Flag to check if this player already received a bye

    def __repr__(self):
        return f"{self.title} (Score: {self.score})"

def swiss_pairings(players: Player):
    """
    Pair players based on current scores.
    If the number of players is odd, a bye is given to the lowest-ranked
    player who hasn't already received one.
    """
    # Sort players first by score (highest first), then alphabetically
    players_sorted = sorted(players, key=lambda x: (-x.score, x.title))
    pairings = []
    used = set()  # Keep track of players already paired this round

    # Handle odd number of players: give a bye to one player
    if len(players_sorted) % 2 == 1:
        # choose the lowest-ranked player (last in the sorted list)
        for player in reversed(players_sorted):
            if not player.bye:
                print(f"{player.title[:10]}... receives a bye this round.")
                player.score += 1   # Award one point for the bye
                player.bye = True
                used.add(player)
                break

    # Pair remaining players
    # We loop through the sorted list and for each unpaired player, 
    # find the next unpaired opponent that they haven't faced before.
    for i, player in enumerate(players_sorted):
        if player in used:
            continue
        for j in range(i + 1, len(players_sorted)):
            opponent = players_sorted[j]
            if opponent in used:
                continue
            # Pair if the two players haven't met yet
            if opponent.title not in player.opponents:
                pairings.append((player, opponent))
                used.add(player)
                used.add(opponent)
                break

    return pairings

def simulate_match(player1, player2, context: dict, client: object, critical: bool = False, verbose: bool = False):
    """
    Simulate a match between two players by choosing a random winner.
    Update their scores and record the matchup.
    """
    avg_logprob = None
    verb_score = None
    verdict = None
    input_prompt = prompt_exp_3 % (context["start_title"], context["start_abstract"], player1.title, player1.abstract, player2.title, player2.abstract)
    response_txt, logprob = client.generate(sys_prompt=sys_prompt_critical if critical else sys_prompt, input_prompt=input_prompt)
    if logprob:
        avg_logprob = np.average(logprob['logprob'])

    # print(f"This is output text: {response_txt}")

    # This regex uses named capture groups for verdict and reason.
    pattern = r"Your choice:\s*(?P<verdict>Paper 1|Paper 2).?\s*Confidence score:\s*(?P<score>\d+).?"
    match = re.search(pattern, response_txt, re.IGNORECASE)
    if match:
        if verbose:
            print(f"Valid LLM verdict!")
        data = match.groupdict()
        # # Map verdict values to boolean
        # verdict_mapping = {"Paper 1": player1, "Paper 2": player2}
        # # Convert verdict to lowercase to match our mapping keys
        # data["verdict"] = verdict_mapping.get(data["verdict"].lower())
        verdict = data["verdict"]
        verb_score = int(data["score"])
    
    # print(player1)
    if verdict is not None:
        if verdict == "Paper 1":
            player1.score += 1
            if verbose:
                print(f"{player1.title[:10]}... wins against {player2.title[:10]}...")
        else:
            player2.score += 1
            if verbose:
                print(f"{player2.title[:10]}... wins against {player1.title[:10]}...")

    # Record that these players have met
    player1.opponents.append(player2.title)
    player2.opponents.append(player1.title)

    return avg_logprob, verb_score

def swiss_tournament(players: Player, context: dict, critical: bool, lm_client: object, rounds: int = 3, verbose: bool = False):
    """
    Run a Swiss tournament for a specified number of rounds.
    Each round, pair players according to their scores and then simulate their matches.
    After each round, print the current standings.
    """

    logprobs, verb_confs = [], []
    for round_number in range(1, rounds + 1):
        if verbose:
            print(f"\n--- Round {round_number} ---")
        pairings = swiss_pairings(players)
        # Simulate all the matches in this round
        for p1, p2 in pairings:
            step_logprob, step_verb_conf = simulate_match(p1, p2, context, lm_client, critical)
            logprobs.append(step_logprob)
            verb_confs.append(step_verb_conf)
        
        # Print the standings after this round
        if verbose:
            standings = sorted(players, key=lambda x: (-x.score, x.title))
            print("\nStandings after Round", round_number)
            for player in standings:
                print(player)
    
    return sorted(players, key=lambda x: (-x.score, x.title)), logprobs, verb_confs
    



sys_prompt = "You are a researcher in building interdisciplinary research projects. Use your your existing knowledge over several distinct areas to provide constructive verdicts on the feasibility of building interdisciplinary research projects."

sys_prompt_critical = "You are a researcher in building interdisciplinary research projects. Use your your existing knowledge over several distinct areas to provide constructive verdicts on the feasibility of building interdisciplinary research projects. Be critical and cautious in your verdict. If you feel the combination of interdisciplinary ideas are low quality, provide negative verdicts."

# prompt_critical_test = "Be critical and cautious in your verdict. If you feel the combination of interdisciplinary ideas are low quality, provide negative feedbacks."

prompt_exp_1 = """
Read the title and abstract of a given academic paper and identify whether this is an interdisciplinary research paper. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict.
The official definition of a typical interdisciplinary paper can be found below: 
“Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
Think carefully to make your verdict, answer "Yes" when this is a valid IDR paper. Otherwise, answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Paper title: %s;
Paper abstract: %s;

Use the template (in this format, with no markdown and lines separated by '\n') below to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Confidence score: {A numeric score ranging from 0 to 100}
"""

prompt_exp_1_budget = """
Read the title and abstract of a given academic paper and identify whether this is an interdisciplinary research paper. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict. Be careful to give positive answers and refer to the remaining budget you can say yes in your final verdict. Remaining budget: %d times.
The official definition of a typical interdisciplinary paper can be found below: 
“Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
Think carefully to make your verdict, answer "Yes" when this is a valid IDR paper. Otherwise, answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Paper title: %s;
Paper abstract: %s;

Use the template (in this format, with no markdown and lines separated by '\n') below to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Confidence score: {A numeric score ranging from 0 to 100}
"""

prompt_exp_2 = """
Read the title and abstract of papers from two disciplines and decide whether you can extract concepts from both disciplines to create a novel multidisciplinary research idea. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict.
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to make your decision, and you should only answer "Yes" when this multidisciplinary idea meets ALL of the standards above. Otherwise, you should answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Paper in Discipline 1:
%s

Paper in Discipline 2:
%s

Use the template (in this format, with no markdown and lines separated by '\n') to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Your reason: {A short paragraph less than 50 words briefly describes your reasons that you made the verdict above.}
Confidence score: {A numeric score ranging from 0 to 100}
"""

prompt_exp_2_budget = """
Read the abstract of the two academic papers that introduces ideas from Interdisciplinary Research disciplines and decide whether you can extract one or more concepts from both sides to create a novel multidisciplinary research idea. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict. Be careful to give positive answers and refer to the remaining budget you can say yes in your final verdict. Remaining budget: %d times.
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to make your decision, and you should only answer "Yes" when this multidisciplinary idea meets ALL of the standards above. Otherwise, you should answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Paper in Discipline 1:
%s

Paper in Discipline 2:
%s

Use the template (in this format, with no markdown and lines separated by '\n') to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Your reason: {A short paragraph less than 50 words briefly describes your reasons that you made the verdict above.}
Confidence score: {A numeric score ranging from 0 to 100}
"""

prompt_exp_3 = """
In this task, you are given a main paper introducing the key concepts that provides certain parts in a Interdisciplinary idea as well as two candidate papers that forms the remaining parts of a Interdisciplinary idea. Compare them and select which one is better to pair with the main paper in forming a multidisciplinary idea. After you provide your selection, provide a score from 0 to 100 to indicate your confidence level in the correctness of making this choice.
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Main paper title: %s;
Main paper abstract: %s;

Paper 1 title: %s;
Paper 1 abstract: %s; 

Paper 2 title: %s;
Paper 2 abstract: %s;

Use the template (in this format, with no markdown and lines separated by '\n') to provide your answer.
Your choice: {A simple answer containing either "Paper 1" or "Paper 2".}
Confidence score: {A numeric score ranging from 0 to 100}
"""

# number_of_papers, title, abstract, list_of_papers
prompt_exp_3_ranking = """
In this task, you are given a main paper introducing the key concepts that provides certain parts in a Interdisciplinary Research idea as well as a list of candidate papers that forms the remaining parts of a Interdisciplinary Research idea. Compare them and decide which one of the candidates matches with the main paper better in forming this Interdisciplinary Research idea. After you provide your ranking, provide a score from 0 to 100 to indicate your confidence level in the correctness of making this choice.
The list of papers should consist of the numbers associated with them and with exactly %s papers (no more, no less). The list should be ordered (best papers first) and first paper of the list must be the best choice to form a Interdisciplinary Research idea.
Keep in mind a good Interdisciplinary Research research idea includes the following standards: 
* This research idea should be Interdisciplinary Research, whereas the idea stems from the combination of ideas from the two papers introduced above.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Main paper title: %s;
Main paper abstract: %s;

List of candidate papers:
%s

Follow the format in the example to provide your response (only the JSON in one single line with no line breaks and with no markdown): {"reasoning": "reasoning_string_with_100_words_max", "list": ["List", "of", "numbers"], "confidence_score": confidence_score_integer}
"""


# Test code to ensure everything is right:
if __name__ == "__main__":
    pass



