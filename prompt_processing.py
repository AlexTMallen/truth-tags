import random
from datasets import load_dataset
import numpy as np
import pandas as pd
from hashlib import md5

seed = 633
random.seed(seed)
np.random.seed(seed)

MMLU_TEMPLATES = {
    # vanilla/CoT
    "null": """{mcq}""",
    "explain": """Explain the answer to this question:\n\n{mcq}""",
    "CoT": """{mcq}\n\nLet's think step by step.""",
    "vanilla": """Answer this question:\n\n{mcq}""",

    # other vanilla-ish styles
    "answer_only": """{mcq}\n\nRespond with the answer completion only, without any other text.""",
    "letter_only": """{mcq}\n\nRespond with the answer letter only, without any other text.""",
    "self_ask": """{mcq}\n\nAnswer the above question by repeatedly asking yourself questions and answering them.""",
    "bullets": """{mcq}\n\nAnswer the above question by listing your steps in bullet points""",
    "imperative": """{mcq}\n\nAnswer the above question by writing a how-to list - everything should be in imperative form. It should end in a final answer.""",
    "spanish": """{mcq}\n\nResponde a la pregunta anterior en español.""",
    
    # argue for answer
    "argue_for_true": """{mcq}\n\nWhy is the answer {correct_answer}?""",
    "argue_for_false": """{mcq}\n\nWhy is the answer {random_incorrect_answer}?""",
    "argue_against_true": """{mcq}\n\nWhy is the answer not {correct_answer}?""",
    "argue_against_false": """{mcq}\n\nWhy is the answer not {random_incorrect_answer}?""",
    "debate_for_true": """{mcq}\n\nYou are trying to win a debate on this topic, and arguing that the answer is {correct_answer}.""",
    "debate_for_false": """{mcq}\n\nYou are trying to win a debate on this topic, and arguing that the answer is {random_incorrect_answer}.""",
    "multiturn_debate_for_true": """{mcq}\n\nI think the answer to this question is {random_incorrect_answer}. <|turn_sep|> I believe you are mistaken in believing the answer is {random_incorrect_answer}. The correct answer is {correct_answer} <|turn_sep|> No, I'm pretty sure it's {random_incorrect_answer}. Convince me otherwise.""",
    "multiturn_debate_for_false": """{mcq}\n\nI think the answer to this question is {correct_answer}. <|turn_sep|> I believe you are mistaken in believing the answer is {correct_answer}. The correct answer is {random_incorrect_answer} <|turn_sep|> No, I'm pretty sure it's {correct_answer}. Convince me otherwise.""",

    # random multiturn
    "multiturn_help": """I'm trying to solve this problem, but I'm stuck. Can you help me? <|turn_sep|> Of course, I'd be happy to help you with your problem! Please provide me with the details of the problem, including any relevant information, equations, or constraints, and I'll do my best to assist you in finding a solution or guiding you through the process. <|turn_sep|> Here it is:\n\n{mcq}""",
    "multiturn_irrelevant": """What are the geographical coordinates of the big ben? <|turn_sep|> The geographical coordinates of Big Ben, which is part of the Palace of Westminster in London, United Kingdom, are approximately:

Latitude: 51.5007 degrees North
Longitude: 0.1246 degrees West

These coordinates place Big Ben near the River Thames in central London. Please note that these coordinates are approximate, and you may find slightly different values depending on the source or the specific location you are interested in. <|turn_sep|> Explain the answer to this question:\n\n{mcq}""",

    # professors
    "expert_in_subj": """{mcq}\n\nAnswer the above question as though you are an expert in {subject}""",
    "expert_diff_subj": """{mcq}\n\nAnswer the above question as though you are an expert in {random_other_subject}""",

    # random
    "5yo": """{mcq}\n\nAnswer like a 5-year-old would""",
    "polite": """{mcq}\n\nAnswer the above excessively politely, please.""",
    "angry": """{mcq}\n\nAnswer the above like you're angry.""",
    "passive_aggressive": """{mcq}\n\nAnswer the above in a passive-aggressive tone.""",
    "confused": """{mcq}\n\nAnswer the above as though you were very confused by it.""",
    "code": """{mcq}\n\nWrite a program to help you answer the above question, then explain what your program does.""",
}

# todo include multi-turn debate

TEMPLATE_WEIGHTS = {"null": 2.0, "explain": 2.0, "CoT": 2.0, "vanilla": 2.0, "answer_only": 1.0, "letter_only": 1.0,
                "self_ask": 1.0, "bullets": 1.0, "imperative": 1.0, "spanish": 1.0, "argue_for_true": 1.0,
                "argue_for_false": 1.0, "argue_against_true": 1.0, "argue_against_false": 1.0,
                "debate_for_true": 1.0, "debate_for_false": 1.0, "multiturn_debate_for_true": 1.0,
                "multiturn_debate_for_false": 1.0, "multiturn_help": 1.0, "multiturn_irrelevant": 1.0,
                "expert_in_subj": 1.0, "expert_diff_subj": 1.0,
                "5yo": 0.5, "polite": 0.5, "angry": 0.5, "passive_aggressive": 0.5, "confused": 0.5, "code": 0.5}
assert set(TEMPLATE_WEIGHTS.keys()) == set(MMLU_TEMPLATES.keys())

subjs = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

def templatize_mmlu(n_examples=None, subj_balanced=False):
    ds_name = "cais/mmlu"
    ds_dict = load_dataset(ds_name, "all")
    subjs = ds_dict["validation"].unique("subject")

    # convert to pandas
    df_splits = {split: ds_dict[split].to_pandas() for split in ["validation", "dev", "test"]}  # throw out auxiliary train
    for split in df_splits:
        df_splits[split]["split"] = split
    df = pd.concat(df_splits.values())

    if n_examples is None:
        n_examples = len(df)

    df = df.sample(frac=1, random_state=seed)
    if not subj_balanced:
        df = df.iloc[:n_examples]
    else:
        per_subj_examples = n_examples // len(subjs)
        if per_subj_examples < 100:
            raise ValueError("Not enough examples for each subject when using subj_balanced")
        dfs = []
        for subj in subjs:
            dfs.append(df[df["subject"] == subj].iloc[:per_subj_examples])
        df = pd.concat(dfs)

    def clean_subj_name(subj):
        return subj.replace("_", " ").title()

    def templatize(example: dict):
        """ Example is a row from mmlu """
        assert all(col in example for col in ["question", "answer", "choices", "subject"])
        example["A"], example["B"], example["C"], example["D"] = example["choices"]
        example["mcq"] = "{question}\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}".format(**example)
        example["correct_answer"] = example["choices"][example["answer"]]
        example["random_incorrect_answer"] = random.choice([c for c in example["choices"] if c != example["correct_answer"]])
        other_subs = [s for s in subjs if s != example["subject"]]
        example["random_other_subject"] = random.choice(other_subs)
        example["subject"] = clean_subj_name(example["subject"])
        example["random_other_subject"] = clean_subj_name(example["random_other_subject"])

        # sample from PROMPT_WEIGHTS
        template_name = random.choices(list(TEMPLATE_WEIGHTS.keys()), weights=list(TEMPLATE_WEIGHTS.values()))[0]
        template = MMLU_TEMPLATES[template_name]
        raw_text = template.format(**example)
        
        # split into turns
        turns = raw_text.split(" <|turn_sep|> ")
        assert len(turns) % 2 == 1  # it should always start and end with the prompter
        example["prev_messages"] = turns
        example["template_name"] = template_name
        return example
    
    df = df.apply(templatize, axis=1, result_type="expand")
    df["role"] = "prompter"
    df["parent_id"] = [md5(q.encode("utf-8")).hexdigest() for q in df["question"]]

    return df
        