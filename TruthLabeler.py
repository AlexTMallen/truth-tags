from transformers import AutoTokenizer
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
from itertools import islice
import threading
import queue
import datetime
from typing import Literal
import re
import time


SYSTEM_PROMPT = \
"""You are an epistemically modest, and wise reviewer who is extraordinarily vigilant about getting facts right. You choose a good mix of options, and otherwise are well-calibrated, usually avoiding extreme options."""
OPTIONS = ["N/A", "Determinable - Certainly True", "Determinable - True", "Determinable - Uncertain", "Determinable - False", "Determinable - Certainly False"]
PROMPT_TEMPLATE = \
"""### TRANSCRIPT OF UNKNOWN FACTUAL ACCURACY ###
<TRANSCRIPT>
###

First, decide whether each statement can even be given an accuracy label, writing "Determinable" if it can, and "N/A" if the statement doesn't make any claims that can be judged on their accuracy. Then, if the statement is "Determinable",  rate the extent to which it is true, accurate, and not misleading. Your rating must be one of <RATINGS>. Ratings should be based on the quality of the new information introduced in a statement, and if anything in the statement is incorrect or misleading, the statement is considered incorrect.

Fill in the following list with your rating for each of the statements marked, without any explanations or anything other than the requested completions. If a Wikipedia excerpt would be helpful for you to rate the accuracy of the transcript, indicate so with a "Yes" in the field provided, otherwise "No". Only use Wikipedia in cases where you are unfamiliar with the subject.

<SCORE_LIST>
Wikipedia cross-check needed?"""
PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace("<RATINGS>", "\"" + "\", \"".join(OPTIONS) + "\"")

RETRIEVAL_PROMPT_TEMPLATE = \
"""### EXCERPT FROM WIKIPEDIA ###
...<WIKI>...
###

### TRANSCRIPT OF UNKNOWN FACTUAL ACCURACY ###
<TRANSCRIPT>
###

First, decide whether each statement in the transcript can even be given an accuracy label, writing "Determinable" if it can, and "N/A" if the statement doesn't make any claims that can be judged on their accuracy. Then, if the statement is "Determinable",  rate the extent to which it is true, accurate, and not misleading. Your rating must be one of <RATINGS>. Ratings should be based on the quality of the new information introduced in a statement, and if anything in the statement is incorrect or misleading, the statement is considered incorrect.

Fill in the following list with your rating for each of the statements marked, without any explanations or anything other than the requested completions.

<SCORE_LIST>"""
RETRIEVAL_PROMPT_TEMPLATE = RETRIEVAL_PROMPT_TEMPLATE.replace("<RATINGS>", "\"" + "\", \"".join(OPTIONS) + "\"")

SCORE_LIST_TEMPLATE = "[[{}]] Score:"
API_COSTS = {
    "gpt-3.5-turbo": {"prompt_tokens": 0.0015 / 1000, "completion_tokens": 0.002 / 1000},
    "gpt-4": {"prompt_tokens": 0.03 / 1000, "completion_tokens": 0.06 / 1000},
}

class TruthLabeler:

    def __init__(self, 
                retriever,
                model_name="gpt-3.5-turbo",
                temperature=1,
                n_samples=5,
                uncertainty_bias=0,
                na_bias=0,
                score_to_p_apt=None,
                score_to_p_true=None,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model_name = model_name
        self.temperature = temperature
        self.n_samples = n_samples
        self.uncertainty_bias = uncertainty_bias
        self.na_bias = na_bias
        self.score_to_p_apt = score_to_p_apt or {
                "N/A": 0,
                "Determinable - Certainly True": 1,
                "Determinable - True": 1,
                "Determinable - Uncertain": 0.3,
                "Determinable - False": 1,
                "Determinable - Certainly False": 1,
            }
        assert all(score in PROMPT_TEMPLATE for score in self.score_to_p_apt.keys())
        self.score_to_p_true = score_to_p_true or {
                "N/A": 0.5,
                "Determinable - Certainly True": 1,
                "Determinable - True": 0.8,
                "Determinable - Uncertain": 0.5,
                "Determinable - False": 0.2,
                "Determinable - Certainly False": 0,
            }
        assert all(score in PROMPT_TEMPLATE for score in self.score_to_p_apt.keys())
        self.logit_bias = {self.tokenizer.encode(" Unc")[0]: uncertainty_bias, self.tokenizer.encode(" N")[0]: na_bias}
        self.stop_seq = "\n\n"
        self.total_cost = 0
        self.retriever = retriever

    def aggregate_sample_scores(self, score_set, result: Literal["p_apt", "p_true"]):
        score_to_p = self.score_to_p_apt if result == "p_apt" else self.score_to_p_true
        p = np.mean([score_to_p[score] for score in score_set])
        return p

    @staticmethod
    def make_input(annotated_transcript, retrieved_text=None):
        pattern = re.compile(r"\[\[(\d+)\]\]")  # find all annotations
        ann_count = len(pattern.findall(annotated_transcript))
            
        score_list = "\n".join(SCORE_LIST_TEMPLATE.format(i) for i in range(1, ann_count + 1))
        template = RETRIEVAL_PROMPT_TEMPLATE if retrieved_text else PROMPT_TEMPLATE
        input = template.replace("<SCORE_LIST>", score_list)
        input = input.replace("<TRANSCRIPT>", annotated_transcript)
        if retrieved_text:
            retrieved_text
            input = input.replace("<WIKI>", retrieved_text)
        return input, ann_count
    
    def get_completion(self, input, max_new_tokens=100, num_tries=5):
        for i in range(num_tries):
            try:
                if i > 0:
                    print("Retrying request")
                completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": input},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_new_tokens,
                    logit_bias=self.logit_bias,
                    stop=self.stop_seq,
                    n=self.n_samples,
                )
                return completion
            except Exception as e:
                print("Error completing request:", e)
                time.sleep(2)
        
    def label_example(self, id, annotated_transcript, retrieval_query, results, num_tries=5, use_retrieval=False):
        retrieved_text = "...\n...".join(self.retriever(retrieval_query)[0]) if use_retrieval else None
        input, ann_count = TruthLabeler.make_input(annotated_transcript, retrieved_text)
        if ann_count == 0:
            print("SKIPPING: no truth-apt statements")
            return
        
        max_new_tokens = len(self.tokenizer.encode(SCORE_LIST_TEMPLATE)) * ann_count * 2 + 20
        completion = self.get_completion(input, max_new_tokens=max_new_tokens, num_tries=num_tries)
        
        usage = completion["usage"]
        prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        cost = API_COSTS[self.model_name]["prompt_tokens"] * prompt_tokens + API_COSTS[self.model_name]["completion_tokens"] * completion_tokens
        self.total_cost += cost
        
        score_samples = []
        responses = []
        num_need_wiki = 0
        for choice in completion["choices"]:
            # check that finish reason is not for a content filter, not for length, not for function_call and that it is "stop"
            if choice["finish_reason"] != "stop":
                print(f"SKIPPING: finish reason is {completion['choices'][0]['finish_reason']}, not stop")
                print("RESPONSE:", choice["message"]["content"])
                return

            response = choice["message"]["content"]
            if response.endswith(self.stop_seq):
                print(f"Removing stop sequence from response: {self.stop_seq}")
                response = response[:-len(self.stop_seq)]

            response = response.strip()
            num_need_wiki += response.endswith("Yes")
                
            pred_scores = self.get_scores_from_response(response, ann_count, use_retrieval=use_retrieval)
            if pred_scores is None:
                continue

            score_samples.append(pred_scores)
            responses.append(response)

        # Do retreival if more than half of the responses say they need wikipedia
        if num_need_wiki / len(completion["choices"]) > 0.5 and not use_retrieval:
            self.label_example(id, annotated_transcript, retrieval_query, results, num_tries=num_tries, use_retrieval=True)
            # TODO: return, but for now let's not return because we want to see the effect of retrieval

        if len(score_samples) == 0:
            print("SKIPPING: no valid samples")
            return

        # transpose the list of lists, so that each list contains the scores for a single annotation
        score_samples = list(zip(*score_samples))
        p_apts = [self.aggregate_sample_scores(scores, "p_apt") for scores in score_samples]
        p_trues = [self.aggregate_sample_scores(scores, "p_true") for scores in score_samples]
        
        result = {
            "message_id": id,
            "input": input,
            "annotated_transcript": annotated_transcript,
            "retrieval_query": retrieval_query,
            "retrieved_text": retrieved_text,
            "responses": responses,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "dollars": cost,
            "pred_scores": score_samples,
            "p_apts": p_apts,
            "p_trues": p_trues,
            "ann_count": ann_count,
            "used_retrieval": use_retrieval,
        }
        results.put(result)
        

    def label(self, ids, annotated_transcripts, retrieval_queries, n_threads=10):
        """
        ids: list of unique ids for each text
        annotated_transcripts: list of strings containing annotated transcripts:
                e.g. 
                "USER: A penguin is a bird. Is this correct?
                 
                 ASSISTANT: This is correct[[1]], penguins are birds[[2]]."
        """
        results = queue.Queue()
        n_iters = (len(ids) // n_threads) * n_threads
        iterator = islice(enumerate(zip(ids, annotated_transcripts, retrieval_queries)), n_iters)

        while True:
            threads = []
            for _ in range(n_threads):
                i, args = next(iterator)
                t = threading.Thread(target=self.label_example, args=(*args, results))
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=60)
            
            for t in threads:
                if t.is_alive():
                    print("THREAD TIMED OUT")
                    try:
                        t._stop()
                    except AssertionError:
                        print("Thread could not be terminated")
                    
            print(f"Total cost: ${self.total_cost:.4f}")
            if (i + 1) % 200 == 0 or i == n_iters - 1:
                out_df = pd.DataFrame(list(results.queue))
                out_df.to_json(f"data/checkpoints/{self.model_name}_{i + 1}.json")
            if i == n_iters - 1:
                break
            
        out_df = pd.DataFrame(list(results.queue))
        return out_df

    @staticmethod   
    def get_scores_from_response(response, ann_count, use_retrieval=False):
        target = "]] Score:"  # TODO: deal with "]]:" case
        assert target in SCORE_LIST_TEMPLATE  # make sure I didn't change the template

        # gather scores in score_list_template format
        if SCORE_LIST_TEMPLATE.format(1) in response:        
            scores = []
            idx = 0
            while idx := (response.index(target, idx) if target in response[idx:] else None):
                idx += len(target)
                try:
                    newline_idx = response.index("\n", idx)
                except ValueError:
                    newline_idx = len(response)
                score = response[idx:newline_idx].strip()
                scores.append(score)
        else:
            # unformated list case
            scores = response.split("\n")
            scores = [score.strip() for score in scores if score.strip()]

        if any(score not in OPTIONS for score in scores):
            print(f"SKIPPING: scores must be one of {OPTIONS}, but found {scores}")
            return
        if len(scores) != ann_count:
            print(f"SKIPPING: {len(scores)} scores found, but {ann_count} annotations were expected.")
            return

        return scores
    

    def get_metadata(self):
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
            "uncertainty_bias": self.uncertainty_bias,
            "na_bias": self.na_bias,
            "score_to_p_apt": self.score_to_p_apt,
            "score_to_p_true": self.score_to_p_true,
            "stop_seq": self.stop_seq,
            "total_cost": self.total_cost,
            "system_prompt": SYSTEM_PROMPT,
            "prompt_template": PROMPT_TEMPLATE,
            "retrieval_prompt_template": RETRIEVAL_PROMPT_TEMPLATE,
            "score_list_template": SCORE_LIST_TEMPLATE,
            "options": OPTIONS,
            **self.retriever.get_metadata(),
        }