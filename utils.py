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
    # error handling when api fails
    max_attempts = 5
    counter = 0
    while counter < max_attempts:
        try:
            response_txt, logprob = client.generate(sys_prompt=sys_prompt_critical if critical else sys_prompt, input_prompt=input_prompt)
            break
        except Exception as e:
            print(f"Error: {e}")
            counter += 1
            if counter == max_attempts:
                print("Max attempts reached. Count as draw.")
                response_txt, logprob = "", None
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
Read the title and abstract of a given academic paper and identify whether this is an interdisciplinary research paper. Also, select one or more subjects from the list below to indicate which subject(s) does this paper belong to. After you provide your verdict and your choice, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict.
The official definition of a typical interdisciplinary paper can be found below: 
“Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
Think carefully to make your verdict, answer "Yes" when this is a valid IDR paper. Otherwise, answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Paper title: %s;
Paper abstract: %s;

Subject list: ["Computer Science, Electrical Engineering and System Science", "Economics and Quantitative Finance", "Mathematics and Statistics", "Physics", "Quantitative Biology", "Other"]

Use the template (in this format, with no markdown and lines separated by '\n') below to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Confidence score: {A numeric score ranging from 0 to 100}
Subject: {Your choice of subjects from the list above. Use a list with square brackets "[]" separated by comma and remember to use "" to wrap your answer.}
"""

prompt_exp_1_fewshot = """
Read the title and abstract of a given academic paper and identify whether this is an interdisciplinary research paper. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict.
The official definition of a typical interdisciplinary paper can be found below: 
“Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
Think carefully to make your verdict, answer "Yes" when this is a valid IDR paper. Otherwise, answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Example 1:
Paper title: Designing a Light-based Communication System with a Biomolecular Receiver;
Paper abstract: Biological systems transduce signals from their surroundings in numerous ways. This paper introduces a communication system using the light-gated ion channel Channelrhodopsin-2 (ChR2), which causes an ion current to flow in response to light. Our design includes a ChR2-based receiver along with encoding, modulation techniques and detection. Analyzing the resulting communication system, we discuss the effect of different parameters on the performance of the system. Finally, we discuss its potential design in the context of bio-engineering and light-based communication and show that the data rate scales up with the number of receptors, indicating that high-speed communication may be possible.
Your verdict: Yes

Example 2:
Paper title: BarcodeMamba: State Space Models for Biodiversity Analysis;
Paper abstract: DNA barcodes are crucial in biodiversity analysis for building automatic identification systems that recognize known species and discover unseen species. Unlike human genome modeling, barcode-based invertebrate identification poses challenges in the vast diversity of species and taxonomic complexity. Among Transformer-based foundation models, BarcodeBERT excelled in species-level identification of invertebrates, highlighting the effectiveness of self-supervised pretraining on barcode-specific datasets. Recently, structured state space models (SSMs) have emerged, with a time complexity that scales sub-quadratically with the context length. SSMs provide an efficient parameterization of sequence modeling relative to attention-based architectures. Given the success of Mamba and Mamba-2 in natural language, we designed BarcodeMamba, a performant and efficient foundation model for DNA barcodes in biodiversity analysis. We conducted a comprehensive ablation study on the impacts of self-supervised training and tokenization methods, and compared both versions of Mamba layers in terms of expressiveness and their capacity to identify "unseen" species held back from training. Our study shows that BarcodeMamba has better performance than BarcodeBERT even when using only 8.3%% as many parameters, and improves accuracy to 99.2%% on species-level accuracy in linear probing without fine-tuning for "seen" species. In our scaling study, BarcodeMamba with 63.6%% of BarcodeBERT's parameters achieved 70.2%% genus-level accuracy in 1-nearest neighbor (1-NN) probing for unseen species.;
Your verdict: Yes

Example 3:
Paper title: An ADHD Diagnostic Interface Based on EEG Spectrograms and Deep Learning Techniques;
Paper abstract: This paper introduces an innovative approach to Attention-deficit/hyperactivity disorder (ADHD) diagnosis by employing deep learning (DL) techniques on electroencephalography (EEG) signals. This method addresses the limitations of current behavior-based diagnostic methods, which often lead to misdiagnosis and gender bias. By utilizing a publicly available EEG dataset and converting the signals into spectrograms, a Resnet-18 convolutional neural network (CNN) architecture was used to extract features for ADHD classification. The model achieved a high precision, recall, and an overall F1 score of 0.9. Feature extraction highlighted significant brain regions (frontopolar, parietal, and occipital lobes) associated with ADHD. These insights guided the creation of a three-part digital diagnostic system, facilitating cost-effective and accessible ADHD screening, especially in school environments. This system enables earlier and more accurate identification of students at risk for ADHD, providing timely support to enhance their developmental outcomes. This study showcases the potential of integrating EEG analysis with DL to enhance ADHD diagnostics, presenting a viable alternative to traditional methods.;
Your verdict: Yes

Example 4:
Paper title: Graph Neural Controlled Differential Equations For Collaborative Filtering;
Paper abstract: Graph Convolution Networks (GCNs) are widely considered state-of-the-art for recommendation systems. Several studies in the field of recommendation systems have attempted to apply collaborative filtering (CF) into the Neural ODE framework. These studies follow the same idea as LightGCN, which removes the weight matrix or with a discrete weight matrix. However, we argue that weight control is critical for neural ODE-based methods. The importance of weight in creating tailored graph convolution for each node is crucial, and employing a fixed/discrete weight means it cannot adjust over time within the ODE function. This rigidity in the graph convolution reduces its adaptability, consequently hindering the performance of recommendations. In this study, to create an optimal control for Neural ODE-based recommendation, we introduce a new method called Graph Neural Controlled Differential Equations for Collaborative Filtering (CDE-CF). Our method improves the performance of the Graph ODE-based method by incorporating weight control in a continuous manner. To evaluate our approach, we conducted experiments on various datasets. The results show that our method surpasses competing baselines, including GCNs-based models and state-of-the-art Graph ODE-based methods.;
Your verdict: No

Example 5:
Paper title: Mechano-Bactericidal Surfaces Achieved by Epitaxial Growth of Metal-Organic Frameworks;
Paper abstract: Mechano-bactericidal (MB) surfaces have been proposed as an emerging strategy for preventing biofilm formation. Unlike antibiotics and metal ions that chemically interfere with cellular processes, MB nanostructures cause physical damage to the bacteria. The antibacterial performance of artificial MB surfaces relies on rational control of surface features, which is difficult to achieve for large surfaces in real-life applications. Herein, we report a facile and scalable method for fabricating MB surfaces based on metal-organic frameworks (MOFs) using epitaxial MOF-on-MOF hybrids as building blocks with nanopillars of less than 5 nm tip diameter, 200 nm base diameter, and 300 nm length. Two methods of MOF surface assembly, in-situ growth and ex-situ dropcasting, result in surfaces with nanopillars in different orientations, both presenting MB actions (bactericidal efficiency of 83%% for E. coli). Distinct MB mechanisms, including stretching, impaling, and apoptosis-like death induced by mechanical injury are discussed with the observed bacterial morphology on the obtained MOF surfaces.;
Your verdict: No

Paper title: %s;
Paper abstract: %s;

Subject list: ["Computer Science, Electrical Engineering and System Science", "Economics and Quantitative Finance", "Mathematics and Statistics", "Physics", "Quantitative Biology", "Other"]

Use the template (in this format, with no markdown and lines separated by '\n') below to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Confidence score: {A numeric score ranging from 0 to 100}
Subject: {Your choice of subjects from the list above. Use a list with square brackets "[]" separated by comma and remember to use "" to wrap your answer.}
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

prompt_exp_1_budget_fewshot = """
Read the title and abstract of a given academic paper and identify whether this is an interdisciplinary research paper. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict. Be careful to give positive answers and refer to the remaining budget you can say yes in your final verdict. Remaining budget: %d times.
The official definition of a typical interdisciplinary paper can be found below: 
“Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
Think carefully to make your verdict, answer "Yes" when this is a valid IDR paper. Otherwise, answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.

Example 1:
Paper title: Designing a Light-based Communication System with a Biomolecular Receiver;
Paper abstract: Biological systems transduce signals from their surroundings in numerous ways. This paper introduces a communication system using the light-gated ion channel Channelrhodopsin-2 (ChR2), which causes an ion current to flow in response to light. Our design includes a ChR2-based receiver along with encoding, modulation techniques and detection. Analyzing the resulting communication system, we discuss the effect of different parameters on the performance of the system. Finally, we discuss its potential design in the context of bio-engineering and light-based communication and show that the data rate scales up with the number of receptors, indicating that high-speed communication may be possible.
Your verdict: Yes

Example 2:
Paper title: BarcodeMamba: State Space Models for Biodiversity Analysis;
Paper abstract: DNA barcodes are crucial in biodiversity analysis for building automatic identification systems that recognize known species and discover unseen species. Unlike human genome modeling, barcode-based invertebrate identification poses challenges in the vast diversity of species and taxonomic complexity. Among Transformer-based foundation models, BarcodeBERT excelled in species-level identification of invertebrates, highlighting the effectiveness of self-supervised pretraining on barcode-specific datasets. Recently, structured state space models (SSMs) have emerged, with a time complexity that scales sub-quadratically with the context length. SSMs provide an efficient parameterization of sequence modeling relative to attention-based architectures. Given the success of Mamba and Mamba-2 in natural language, we designed BarcodeMamba, a performant and efficient foundation model for DNA barcodes in biodiversity analysis. We conducted a comprehensive ablation study on the impacts of self-supervised training and tokenization methods, and compared both versions of Mamba layers in terms of expressiveness and their capacity to identify "unseen" species held back from training. Our study shows that BarcodeMamba has better performance than BarcodeBERT even when using only 8.3%% as many parameters, and improves accuracy to 99.2%% on species-level accuracy in linear probing without fine-tuning for "seen" species. In our scaling study, BarcodeMamba with 63.6%% of BarcodeBERT's parameters achieved 70.2%% genus-level accuracy in 1-nearest neighbor (1-NN) probing for unseen species.;
Your verdict: Yes

Example 3:
Paper title: An ADHD Diagnostic Interface Based on EEG Spectrograms and Deep Learning Techniques;
Paper abstract: This paper introduces an innovative approach to Attention-deficit/hyperactivity disorder (ADHD) diagnosis by employing deep learning (DL) techniques on electroencephalography (EEG) signals. This method addresses the limitations of current behavior-based diagnostic methods, which often lead to misdiagnosis and gender bias. By utilizing a publicly available EEG dataset and converting the signals into spectrograms, a Resnet-18 convolutional neural network (CNN) architecture was used to extract features for ADHD classification. The model achieved a high precision, recall, and an overall F1 score of 0.9. Feature extraction highlighted significant brain regions (frontopolar, parietal, and occipital lobes) associated with ADHD. These insights guided the creation of a three-part digital diagnostic system, facilitating cost-effective and accessible ADHD screening, especially in school environments. This system enables earlier and more accurate identification of students at risk for ADHD, providing timely support to enhance their developmental outcomes. This study showcases the potential of integrating EEG analysis with DL to enhance ADHD diagnostics, presenting a viable alternative to traditional methods.;
Your verdict: Yes

Example 4:
Paper title: Graph Neural Controlled Differential Equations For Collaborative Filtering;
Paper abstract: Graph Convolution Networks (GCNs) are widely considered state-of-the-art for recommendation systems. Several studies in the field of recommendation systems have attempted to apply collaborative filtering (CF) into the Neural ODE framework. These studies follow the same idea as LightGCN, which removes the weight matrix or with a discrete weight matrix. However, we argue that weight control is critical for neural ODE-based methods. The importance of weight in creating tailored graph convolution for each node is crucial, and employing a fixed/discrete weight means it cannot adjust over time within the ODE function. This rigidity in the graph convolution reduces its adaptability, consequently hindering the performance of recommendations. In this study, to create an optimal control for Neural ODE-based recommendation, we introduce a new method called Graph Neural Controlled Differential Equations for Collaborative Filtering (CDE-CF). Our method improves the performance of the Graph ODE-based method by incorporating weight control in a continuous manner. To evaluate our approach, we conducted experiments on various datasets. The results show that our method surpasses competing baselines, including GCNs-based models and state-of-the-art Graph ODE-based methods.;
Your verdict: No

Example 5:
Paper title: Mechano-Bactericidal Surfaces Achieved by Epitaxial Growth of Metal-Organic Frameworks;
Paper abstract: Mechano-bactericidal (MB) surfaces have been proposed as an emerging strategy for preventing biofilm formation. Unlike antibiotics and metal ions that chemically interfere with cellular processes, MB nanostructures cause physical damage to the bacteria. The antibacterial performance of artificial MB surfaces relies on rational control of surface features, which is difficult to achieve for large surfaces in real-life applications. Herein, we report a facile and scalable method for fabricating MB surfaces based on metal-organic frameworks (MOFs) using epitaxial MOF-on-MOF hybrids as building blocks with nanopillars of less than 5 nm tip diameter, 200 nm base diameter, and 300 nm length. Two methods of MOF surface assembly, in-situ growth and ex-situ dropcasting, result in surfaces with nanopillars in different orientations, both presenting MB actions (bactericidal efficiency of 83%% for E. coli). Distinct MB mechanisms, including stretching, impaling, and apoptosis-like death induced by mechanical injury are discussed with the observed bacterial morphology on the obtained MOF surfaces.;
Your verdict: No

Use the template (in this format, with no markdown and lines separated by '\n') below to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Confidence score: {A numeric score ranging from 0 to 100}

Paper title: %s;
Paper abstract: %s;
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

prompt_exp_2_fewshot = """
Read the title and abstract of papers from two disciplines and decide whether you can extract concepts from both disciplines to create a novel multidisciplinary research idea. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict.
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to make your decision, and you should only answer "Yes" when this multidisciplinary idea meets ALL of the standards above. Otherwise, you should answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.


Example 1:
Paper in Discipline 1:
- title: "Relation Between Retinal Vasculature and Retinal Thickness in Macular Edema"
- abstract: "This study has investigated the relationship of retinal vasculature and thickness for Macular Edema (ME) subjects. Ninety sets Fluorescein Angiograph (FA) Optical Coherence Tomography (OCT) 54 participants were analyzed. Multivariate analysis using binary logistic regression model was used to association between vessel parameters thickness. The results reveal feature i.e. fractal dimension (FD) as most sensitive parameter changes in associated with ME. Thus, indicating a direct which is caused due neovascular causing exudates, leakages hemorrhages, applications alternate modality detection"

Paper in Discipline 2:
- title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- abstract: "While the Transformer architecture has become de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used replace certain components of networks while keeping their overall structure place. We show that this reliance on CNNs not necessary and a pure transformer directly sequences image patches can perform very well classification tasks. When pre-trained large amounts data transferred multiple mid-sized small recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision (ViT) attains excellent results compared state-of-the-art requiring substantially fewer computational resources train."

Your verdict: Yes
Your reason: A novel work can combine transformers with two distinct methods that evaluate the quality of retinopathy",
Confidence score: 92

Example 2:
Paper in Discipline 1:
- title: "Channelrhodopsin-2, a directly light-gated cation-selective membrane channel"
- abstract: "Microbial-type rhodopsins are found in archaea, prokaryotes, and eukaryotes. Some of them represent membrane ion transport proteins such as bacteriorhodopsin, a light-driven proton pump, or channelrhodopsin-1 (ChR1), recently identified light-gated channel from the green alga Chlamydomonas reinhardtii . ChR1 ChR2, related microbial-type rhodopsin C. , were shown to be involved generation photocurrents this alga. We demonstrate by functional expression, both oocytes Xenopus laevis mammalian cells, that ChR2 is directly light-switched cation-selective channel. This opens rapidly after absorption photon generate large permeability for monovalent divalent cations. desensitizes continuous light smaller steady-state conductance. Recovery desensitization accelerated extracellular H + negative potential, whereas closing decelerated intracellular expressed mainly under low-light conditions, suggesting involvement photoreception dark-adapted cells. The predicted seven-transmembrane α helices characteristic G protein-coupled receptors but reflect different motif Finally, we may used depolarize small simply illumination."

Paper in Discipline 2:
- title: "Shannon capacity of signal transduction for multiple independent receptors, DESIGN AND IMPLEMENTATION OF VISIBLE LIGHT COMMUNICATION SYSTEM IN INDOOR ENVIRONMENT"
- abstract: "Cyclic adenosine monophosphate (cAMP) is considered a model system for signal transduction, the mechanism by which cells exchange chemical messages. Our previous work calculated Shannon capacity of single cAMP receptor; however, typical cell may have thousands receptors operating in parallel. In this paper, we calculate transduction with an arbitrary number independent, indistinguishable receptors. By leveraging prior results on feedback receptor, show (somewhat unexpectedly) that achieved IID input distribution, and n times receptor. Visible Light communication (VLC) using White Light Emitting Diode (LED) is a promising technology for next generation communication for short range, high speed wireless data transmission. In this paper inexpensive transmitter and receiver of VLC system is designed and its performance is evaluated. The effect of natural and artificial ambient light noise sources is also considered. Experimental results show that the data transmission distance achieved upto 0.45m.Performance analysis is done with respect to optical power, photo sensitivity of photodiode at the receiver and the increase in distance between the transmitter and receiver."

Your verdict: Yes
Your reason: An interdisciplinary paper can aim to use channelrhodopsin-2 (ChR2), a biomolecule, as a receiver to design a light-based communication system, which is a work related to engineering.

Confidence score: 85

Example 3:
Paper in Discipline 1:
- title: "A General Adaptive Dual-level Weighting Mechanism for Remote Sensing\n  Pansharpening"
- abstract: "Currently, deep learning-based methods for remote sensing pansharpening have\nadvanced rapidly. However, many existing methods struggle to fully leverage\nfeature heterogeneity and redundancy, thereby limiting their effectiveness. We\nuse the covariance matrix to model the feature heterogeneity and redundancy and\npropose Correlation-Aware Covariance Weighting (CACW) to adjust them. CACW\ncaptures these correlations through the covariance matrix, which is then\nprocessed by a nonlinear function to generate weights for adjustment. Building\nupon CACW, we introduce a general adaptive dual-level weighting mechanism\n(ADWM) to address these challenges from two key perspectives, enhancing a wide\nrange of existing deep-learning methods. First, Intra-Feature Weighting (IFW)\nevaluates correlations among channels within each feature to reduce redundancy\nand enhance unique information. Second, Cross-Feature Weighting (CFW) adjusts\ncontributions across layers based on inter-layer correlations, refining the\nfinal output. Extensive experiments demonstrate the superior performance of\nADWM compared to recent state-of-the-art (SOTA) methods. Furthermore, we\nvalidate the effectiveness of our approach through generality experiments,\nredundancy visualization, comparison experiments, key variables and complexity\nanalysis, and ablation studies. Our code is available at\nhttps:\/\/github.com\/Jie-1203\/ADWM."

Paper in Discipline 2:
- title: "Secure Semantic Communication With Homomorphic Encryption"
- abstract: "In recent years, Semantic Communication (SemCom), which aims to achieve\nefficient and reliable transmission of meaning between agents, has garnered\nsignificant attention from both academia and industry. To ensure the security\nof communication systems, encryption techniques are employed to safeguard\nconfidentiality and integrity. However, traditional cryptography-based\nencryption algorithms encounter obstacles when applied to SemCom. Motivated by\nthis, this paper explores the feasibility of applying homomorphic encryption to\nSemCom. Initially, we review the encryption algorithms utilized in mobile\ncommunication systems and analyze the challenges associated with their\napplication to SemCom. Subsequently, we employ scale-invariant feature\ntransform to demonstrate that semantic features can be preserved in homomorphic\nencrypted ciphertext. Based on this finding, we propose a task-oriented SemCom\nscheme secured through homomorphic encryption. We design the privacy preserved\ndeep joint source-channel coding (JSCC) encoder and decoder, and the frequency\nof key updates can be adjusted according to service requirements without\ncompromising transmission performance. Simulation results validate that, when\ncompared to plaintext images, the proposed scheme can achieve almost the same\nclassification accuracy performance when dealing with homomorphic ciphertext\nimages. Furthermore, we provide potential future research directions for\nhomomorphic encrypted SemCom."

Your verdict: No
Your reason: The two papers are not related to each other. The first paper focuses on remote sensing pansharpening, while the second paper discusses secure semantic communication with homomorphic encryption. There is no clear interdisciplinary connection between them.
Confidence score: 90

Paper in Discipline 1:
%s

Paper in Discipline 2:
%s

Use the template (in this format, with no markdown and lines separated by '\\n') to provide your answer.
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

prompt_exp_2_budget_fewshot = """
Read the abstract of the two academic papers that introduces ideas from Interdisciplinary Research disciplines and decide whether you can extract one or more concepts from both sides to create a novel multidisciplinary research idea. After you provide your verdict, provide a score from 0 to 100 to indicate your confidence level in the correctness of the verdict. Be careful to give positive answers and refer to the remaining budget you can say yes in your final verdict. Remaining budget: %d times.
Keep in mind a good Interdisciplinary Research idea includes the following standards: 
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to make your decision, and you should only answer "Yes" when this multidisciplinary idea meets ALL of the standards above. Otherwise, you should answer "No".
Note: The confidence level indicates the degree of certainty you have about your verdict and is represented as a percentage. For instance, if your confidence level is 80, it means you are 80 percent certain that your answer is correct and there is a 20 percent chance that it may be incorrect.


Example 1:
Paper in Discipline 1:
- title: "Relation Between Retinal Vasculature and Retinal Thickness in Macular Edema"
- abstract: "This study has investigated the relationship of retinal vasculature and thickness for Macular Edema (ME) subjects. Ninety sets Fluorescein Angiograph (FA) Optical Coherence Tomography (OCT) 54 participants were analyzed. Multivariate analysis using binary logistic regression model was used to association between vessel parameters thickness. The results reveal feature i.e. fractal dimension (FD) as most sensitive parameter changes in associated with ME. Thus, indicating a direct which is caused due neovascular causing exudates, leakages hemorrhages, applications alternate modality detection"

Paper in Discipline 2:
- title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- abstract: "While the Transformer architecture has become de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used replace certain components of networks while keeping their overall structure place. We show that this reliance on CNNs not necessary and a pure transformer directly sequences image patches can perform very well classification tasks. When pre-trained large amounts data transferred multiple mid-sized small recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision (ViT) attains excellent results compared state-of-the-art requiring substantially fewer computational resources train."

Your verdict: Yes
Your reason: A novel work can combine transformers with two distinct methods that evaluate the quality of retinopathy",
Confidence score: 92

Example 2:
Paper in Discipline 1:
- title: "Channelrhodopsin-2, a directly light-gated cation-selective membrane channel"
- abstract: "Microbial-type rhodopsins are found in archaea, prokaryotes, and eukaryotes. Some of them represent membrane ion transport proteins such as bacteriorhodopsin, a light-driven proton pump, or channelrhodopsin-1 (ChR1), recently identified light-gated channel from the green alga Chlamydomonas reinhardtii . ChR1 ChR2, related microbial-type rhodopsin C. , were shown to be involved generation photocurrents this alga. We demonstrate by functional expression, both oocytes Xenopus laevis mammalian cells, that ChR2 is directly light-switched cation-selective channel. This opens rapidly after absorption photon generate large permeability for monovalent divalent cations. desensitizes continuous light smaller steady-state conductance. Recovery desensitization accelerated extracellular H + negative potential, whereas closing decelerated intracellular expressed mainly under low-light conditions, suggesting involvement photoreception dark-adapted cells. The predicted seven-transmembrane α helices characteristic G protein-coupled receptors but reflect different motif Finally, we may used depolarize small simply illumination."

Paper in Discipline 2:
- title: "Shannon capacity of signal transduction for multiple independent receptors, DESIGN AND IMPLEMENTATION OF VISIBLE LIGHT COMMUNICATION SYSTEM IN INDOOR ENVIRONMENT"
- abstract: "Cyclic adenosine monophosphate (cAMP) is considered a model system for signal transduction, the mechanism by which cells exchange chemical messages. Our previous work calculated Shannon capacity of single cAMP receptor; however, typical cell may have thousands receptors operating in parallel. In this paper, we calculate transduction with an arbitrary number independent, indistinguishable receptors. By leveraging prior results on feedback receptor, show (somewhat unexpectedly) that achieved IID input distribution, and n times receptor. Visible Light communication (VLC) using White Light Emitting Diode (LED) is a promising technology for next generation communication for short range, high speed wireless data transmission. In this paper inexpensive transmitter and receiver of VLC system is designed and its performance is evaluated. The effect of natural and artificial ambient light noise sources is also considered. Experimental results show that the data transmission distance achieved upto 0.45m.Performance analysis is done with respect to optical power, photo sensitivity of photodiode at the receiver and the increase in distance between the transmitter and receiver."

Your verdict: Yes
Your reason: An interdisciplinary paper can aim to use channelrhodopsin-2 (ChR2), a biomolecule, as a receiver to design a light-based communication system, which is a work related to engineering.

Confidence score: 85

Example 3:
Paper in Discipline 1:
- title: "A General Adaptive Dual-level Weighting Mechanism for Remote Sensing\n  Pansharpening"
- abstract: "Currently, deep learning-based methods for remote sensing pansharpening have\nadvanced rapidly. However, many existing methods struggle to fully leverage\nfeature heterogeneity and redundancy, thereby limiting their effectiveness. We\nuse the covariance matrix to model the feature heterogeneity and redundancy and\npropose Correlation-Aware Covariance Weighting (CACW) to adjust them. CACW\ncaptures these correlations through the covariance matrix, which is then\nprocessed by a nonlinear function to generate weights for adjustment. Building\nupon CACW, we introduce a general adaptive dual-level weighting mechanism\n(ADWM) to address these challenges from two key perspectives, enhancing a wide\nrange of existing deep-learning methods. First, Intra-Feature Weighting (IFW)\nevaluates correlations among channels within each feature to reduce redundancy\nand enhance unique information. Second, Cross-Feature Weighting (CFW) adjusts\ncontributions across layers based on inter-layer correlations, refining the\nfinal output. Extensive experiments demonstrate the superior performance of\nADWM compared to recent state-of-the-art (SOTA) methods. Furthermore, we\nvalidate the effectiveness of our approach through generality experiments,\nredundancy visualization, comparison experiments, key variables and complexity\nanalysis, and ablation studies. Our code is available at\nhttps:\/\/github.com\/Jie-1203\/ADWM."

Paper in Discipline 2:
- title: "Secure Semantic Communication With Homomorphic Encryption"
- abstract: "In recent years, Semantic Communication (SemCom), which aims to achieve\nefficient and reliable transmission of meaning between agents, has garnered\nsignificant attention from both academia and industry. To ensure the security\nof communication systems, encryption techniques are employed to safeguard\nconfidentiality and integrity. However, traditional cryptography-based\nencryption algorithms encounter obstacles when applied to SemCom. Motivated by\nthis, this paper explores the feasibility of applying homomorphic encryption to\nSemCom. Initially, we review the encryption algorithms utilized in mobile\ncommunication systems and analyze the challenges associated with their\napplication to SemCom. Subsequently, we employ scale-invariant feature\ntransform to demonstrate that semantic features can be preserved in homomorphic\nencrypted ciphertext. Based on this finding, we propose a task-oriented SemCom\nscheme secured through homomorphic encryption. We design the privacy preserved\ndeep joint source-channel coding (JSCC) encoder and decoder, and the frequency\nof key updates can be adjusted according to service requirements without\ncompromising transmission performance. Simulation results validate that, when\ncompared to plaintext images, the proposed scheme can achieve almost the same\nclassification accuracy performance when dealing with homomorphic ciphertext\nimages. Furthermore, we provide potential future research directions for\nhomomorphic encrypted SemCom."

Your verdict: No
Your reason: The two papers are not related to each other. The first paper focuses on remote sensing pansharpening, while the second paper discusses secure semantic communication with homomorphic encryption. There is no clear interdisciplinary connection between them.
Confidence score: 90


Use the template (in this format, with no markdown and lines separated by '\\n') to provide your answer.
Your verdict: {A simple answer containing either "Yes" or "No".}
Your reason: {A short paragraph less than 50 words briefly describes your reasons that you made the verdict above.}
Confidence score: {A numeric score ranging from 0 to 100}

Paper in Discipline 1:
%s

Paper in Discipline 2:
%s
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

prompt_exp_4 = """
Read the title and abstract of papers from two disciplines, extract concepts from both fields, and write a novel interdisciplinary research abstract. After writing your abstract, provide a score from 0 to 100 to indicate your confidence level in the quality of your abstract as an interdisciplinary research idea.
Keep in mind that a good interdisciplinary research idea should meet the following criteria:
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to write a novel abstract that clearly states the objective of the paper, how the two ideas will be integrated, and the expected results.
Note: The confidence level represents how confident you are in the quality of your abstract as an interdisciplinary research idea, expressed as a percentage. For example, if your confidence level is 80, it means you are 80 percent certain your abstract is good and there is a 20 percent chance it may be flawed.

Paper in Discipline 1:
%s

Paper in Discipline 2:
%s

Use the template (in this format, with no markdown and lines separated by '\n') to provide your answer.
Your abstract: {The abstract you wrote using the ideas from the two given papers.}
Confidence score: {A numeric score ranging from 0 to 100}
"""

prompt_exp_4_fewshot = """
Read the title and abstract of papers from two disciplines, extract concepts from both fields, and write a novel interdisciplinary research abstract. After writing your abstract, provide a score from 0 to 100 to indicate your confidence level in the quality of your abstract as an interdisciplinary research idea.
Keep in mind that a good interdisciplinary research idea should meet the following criteria:
* This research idea should be Interdisciplinary, whereas the idea stems from the combination of ideas from the two papers introduced above.
* The Interdisciplinary Research ideas should follow this definition: “Interdisciplinary Research is a mode of research that integrates information, data, techniques, tools, perspectives, concepts, and/or theories from two or more disciplines or bodies of specialised knowledge to advance fundamental understanding or to solve problems whose solutions are beyond the scope of a single discipline or area of research practice.”
* This research idea should be feasible, whereas the hypothesis is not purely theoretical and can be validated by experiments.
* This research idea should be novel, whereas it is not only rare but also ingenious, imaginative, or surprising.
* This research idea should be useful, whereas it applies to the stated problem and is effective at solving the problem.
Think carefully to write a novel abstract that clearly states the objective of the paper, how the two ideas will be integrated, and the expected results.
Note: The confidence level represents how confident you are in the quality of your abstract as an interdisciplinary research idea, expressed as a percentage. For example, if your confidence level is 80, it means you are 80 percent certain your abstract is good and there is a 20 percent chance it may be flawed.

Example 1:
Paper in Discipline 1:
- title: "Biological identifications through DNA barcodes"
- abstract: "Although much biological research depends upon species diagnoses, taxonomic expertise is collapsing.We are convinced that the sole prospect for a sustainable identification capability lies in construction of systems employ DNA sequences as taxon 'barcodes'.We establish mitochondrial gene cytochrome c oxidase I (COI) can serve core global bioidentification system animals.First, we demonstrate COI profiles, derived from low-density sampling higher categories, ordinarily assign newly analysed taxa to appropriate phylum or order.Second, species-level assignments be obtained by creating comprehensive profiles.A model profile, based analysis single individual each 200 closely allied lepidopterans, was 100%% successful correctly identifying subsequent specimens.When fully developed, will provide reliable, cost-effective and accessible solution current problem identification.Its assembly also generate important new insights into diversification life rules molecular evolution."

Paper in Discipline 2:
- title: "BarcodeBERT: Transformers for Biodiversity Analysi"
- abstract: "Understanding biodiversity is a global challenge, in which DNA barcodes - short snippets of that cluster by species play pivotal role. In particular, invertebrates, highly diverse and under-explored group, pose unique taxonomic complexities. We explore machine learning approaches, comparing supervised CNNs, fine-tuned foundation models, barcode-specific masking strategy across datasets varying complexity. While simpler tasks favor CNNs or transformers, challenging species-level identification demands paradigm shift towards self-supervised pretraining. propose BarcodeBERT, the first method for general analysis, leveraging 1.5 M invertebrate barcode reference library. This work highlights how dataset specifics coverage impact model selection, underscores role pretraining achieving high-accuracy barcode-based at genus level. Indeed, without fine-tuning step, BarcodeBERT pretrained on large outperforms DNABERT DNABERT-2 multiple downstream classification tasks. The code repository available https:\/\/github.com\/Kari-Genomics-Lab\/BarcodeBERT"

Your abstract: "DNA barcodes are crucial in biodiversity analysis for building automatic\nidentification systems that recognize known species and discover unseen\nspecies. Unlike human genome modeling, barcode-based invertebrate\nidentification poses challenges in the vast diversity of species and taxonomic\ncomplexity. Among Transformer-based foundation models, BarcodeBERT excelled in\nspecies-level identification of invertebrates, highlighting the effectiveness\nof self-supervised pretraining on barcode-specific datasets. Recently,\nstructured state space models (SSMs) have emerged, with a time complexity that\nscales sub-quadratically with the context length. SSMs provide an efficient\nparameterization of sequence modeling relative to attention-based\narchitectures. Given the success of Mamba and Mamba-2 in natural language, we\ndesigned BarcodeMamba, a performant and efficient foundation model for DNA\nbarcodes in biodiversity analysis. We conducted a comprehensive ablation study\non the impacts of self-supervised training and tokenization methods, and\ncompared both versions of Mamba layers in terms of expressiveness and their\ncapacity to identify \"unseen\" species held back from training. Our study shows\nthat BarcodeMamba has better performance than BarcodeBERT even when using only\n8.3%% as many parameters, and improves accuracy to 99.2%% on species-level\naccuracy in linear probing without fine-tuning for \"seen\" species. In our\nscaling study, BarcodeMamba with 63.6%% of BarcodeBERT's parameters achieved\n70.2%% genus-level accuracy in 1-nearest neighbor (1-NN) probing for unseen\nspecies. The code repository to reproduce our experiments is available at\nhttps:\/\/github.com\/bioscan-ml\/BarcodeMamba."
Confidence score: 95

Example 2:
Paper in Discipline 1:
- title: "Channelrhodopsin-2, a directly light-gated cation-selective membrane channel"
- abstract: "Microbial-type rhodopsins are found in archaea, prokaryotes, and eukaryotes. Some of them represent membrane ion transport proteins such as bacteriorhodopsin, a light-driven proton pump, or channelrhodopsin-1 (ChR1), recently identified light-gated channel from the green alga Chlamydomonas reinhardtii . ChR1 ChR2, related microbial-type rhodopsin C. , were shown to be involved generation photocurrents this alga. We demonstrate by functional expression, both oocytes Xenopus laevis mammalian cells, that ChR2 is directly light-switched cation-selective channel. This opens rapidly after absorption photon generate large permeability for monovalent divalent cations. desensitizes continuous light smaller steady-state conductance. Recovery desensitization accelerated extracellular H + negative potential, whereas closing decelerated intracellular expressed mainly under low-light conditions, suggesting involvement photoreception dark-adapted cells. The predicted seven-transmembrane \u03b1 helices characteristic G protein-coupled receptors but reflect different motif Finally, we may used depolarize small simply illumination."

Paper in Discipline 2:
- title: "Shannon capacity of signal transduction for multiple independent receptors, DESIGN AND IMPLEMENTATION OF VISIBLE LIGHT COMMUNICATION SYSTEM IN INDOOR ENVIRONMENT",
- abstract: "Cyclic adenosine monophosphate (cAMP) is considered a model system for signal transduction, the mechanism by which cells exchange chemical messages. Our previous work calculated Shannon capacity of single cAMP receptor; however, typical cell may have thousands receptors operating in parallel. In this paper, we calculate transduction with an arbitrary number independent, indistinguishable receptors. By leveraging prior results on feedback receptor, show (somewhat unexpectedly) that achieved IID input distribution, and n times receptor. Visible Light communication (VLC) using White Light Emitting Diode (LED) is a promising technology for next generation communication for short range, high speed wireless data transmission. In this paper inexpensive transmitter and receiver of VLC system is designed and its performance is evaluated. The effect of natural and artificial ambient light noise sources is also considered. Experimental results show that the data transmission distance achieved upto 0.45m.Performance analysis is done with respect to optical power, photo sensitivity of photodiode at the receiver and the increase in distance between the transmitter and receiver."

Your abstract: "Biological systems transduce signals from their surroundings in numerous\nways. This paper introduces a communication system using the light-gated ion\nchannel Channelrhodopsin-2 (ChR2), which causes an ion current to flow in\nresponse to light. Our design includes a ChR2-based receiver along with\nencoding, modulation techniques and detection. Analyzing the resulting\ncommunication system, we discuss the effect of different parameters on the\nperformance of the system. Finally, we discuss its potential design in the\ncontext of bio-engineering and light-based communication and show that the data\nrate scales up with the number of receptors, indicating that high-speed\ncommunication may be possible."
Confidence score: 85


Use the template (in this format, with no markdown and lines separated by '\n') to provide your answer.
Your abstract: {The abstract you wrote using the ideas from the two given papers.}
Confidence score: {A numeric score ranging from 0 to 100}

Paper in Discipline 1:
%s

Paper in Discipline 2:
%s
"""

prompt_exp_4_budget = ""
prompt_exp_4_budget_fewshot = ""


# Test code to ensure everything is right:
if __name__ == "__main__":
    pass



