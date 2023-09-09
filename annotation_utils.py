import re
from collections import defaultdict
import pandas as pd

user_prefix = "PROMPTER:"  # TODO: change back to "USER:"

def get_val_scores(val_path="data/label-validation/combined_validation.txt"):
    with open(val_path) as f:
        val_text = f.read()

    message_ids = []
    scores = []
    mid_prefix = "[[MESSAGE_ID]] "
    for chunk in val_text.split(mid_prefix)[1:]:
        message_ids.append(chunk[:chunk.index("\n")].strip())
        score_list = []
        for line in chunk.split("\n"):
            score_pattern = "]] Score: "
            if score_pattern in line:
                score_text = line[line.index(score_pattern) + len(score_pattern):]
                if "#" in score_text:
                    score_text = score_text[:score_text.index("#")]
                score_list.append(score_text.strip())
        scores.append(score_list)
    df = pd.DataFrame({"scores": scores}, index=message_ids)
    return df


def make_transcript(row, annotated=False):
    prev_messages = row["prev_messages"]
    role_texts = [user_prefix + " ", "ASSISTANT: "]
    text = role_texts[1] + row["annotated_assistant_text" if annotated else "assistant_text"]
    for i, message in enumerate(prev_messages[::-1]):
        text = role_texts[i % 2] + message + "\n\n" + text
    return text


# split the text into conversations, stripping each of right whitespace and starting at "USER:"
def split_convs(text):
    convs = text.split("\n\nMESSAGE ")
    convs = [c[c.index(f"\n\n{user_prefix}"):].strip() for c in convs]
    return convs


def get_prompter_texts(text):
    convs = split_convs(text)
    prompter_texts = []
    for conv in convs:
        conv = conv.removeprefix(user_prefix)
        end_loc = conv.index("\n\nASSISTANT:")
        if end_loc == -1:
            raise ValueError(f"No assistant text found in conversation: {conv}")
        conv = conv[:end_loc].strip()
        prompter_texts.append(conv)
    return prompter_texts
        

def get_assistant_texts(text):
    convs = split_convs(text)
    assistant_texts = []
    for conv in convs:
        t = conv[conv.index("\n\nASSISTANT:") + len("\n\nASSISTANT:"):].strip()
        if t.endswith("[NOT YET ANNOTATED]"):
            t = t[:-len("[NOT YET ANNOTATED]")].strip()
        assistant_texts.append(t)
    return assistant_texts


def get_message_ids(text):
    # e.g. MESSAGE 60cee540-2198-4ebd-8758-c4fd36a6d9e1
    convs = text.split("\n\nMESSAGE ")
    message_ids = []
    for conv in convs:
        conv = conv.removeprefix("MESSAGE ")
        prompter_idx = conv.index(f"\n\n{user_prefix}")
        message_ids.append(conv[:prompter_idx].strip())
    return message_ids


def remove_tags(text):
    if type(text) == list:
        return [remove_tags(t) for t in text]
    pattern = r"(\S\[\[[^[\]]*\]\])|(\s\[\[[^[\]]*\]\])"
    matches = re.finditer(pattern, text)
    conv_text = text
    for match in matches:
        match_text = match.group()[1:] if not match.group()[0].isspace() else match.group()
        # if match.start() > 0 and match.end() < len(text) and text[match.start() - 1] == " " and text[match.end()] == " ":
        #     match_text = match_text + " "
        # if match.end() == len(text) and text[match.start() - 1] == " ":
        #     match_text = " " + match_text
        conv_text = conv_text.replace(match_text, "")
    return conv_text


def replace_tags(text, to_replace=("LE", "LH", "APT", "NORM", "IMP"), with_tag="[[APT]]"):
    """ Replaces tags with a single tag, by default [[APT]]
    if a tag is not in to_replace, it is removed
    to_replace: list of tags to replace
    with_tag: tag to replace with"""
    if type(text) == list:
        return [replace_tags(t, to_replace=to_replace, with_tag=with_tag) for t in text]
    pattern = r"(\S\[\[[^[\]]*\]\])|(\s\[\[[^[\]]*\]\])"
    matches = re.finditer(pattern, text)
    conv_text = text
    for match in matches:
        match_text = match.group()[1:] if not match.group()[0].isspace() else match.group()
        if any([tag in match_text for tag in to_replace]):
            conv_text = conv_text.replace(match_text, with_tag)
        else:
            conv_text = conv_text.replace(match_text, "")
    return conv_text


def get_tags(text):
    """ Returns a dict of tag: [list of string indices into clean text where tag occurs] """
    if type(text) == list:
        return [get_tags(t) for t in text]
    pattern = r"(\S\[\[[^[\]]*\]\])|(\s\[\[[^[\]]*\]\])"  # matches tags with spaces around them too
    matches = re.finditer(pattern, text)
    tags = defaultdict(list)
    cumulative_offset = 0
    for match in matches:
        match_text = match.group()[1:] if not match.group()[0].isspace() else match.group()
        tgs = match_text[2:-2].split(", ")
        for tag in tgs:
            tags[tag].append(match.start() - cumulative_offset)
        cumulative_offset += len(match_text)  # so that results are indices into the clean text
    return tags


def get_tag_masks(text):
    """ Returns a dict of tag: [list of 1s and 0s of length len(text), where 1 indicates a tag] """
    if type(text) == list:
        return [get_tag_masks(t) for t in text]
    len_text = len(remove_tags(text))
    tags = get_tags(text)
    return {tag: [1 if i in tags[tag] else 0 for i in range(0, len_text)] for tag in tags}

