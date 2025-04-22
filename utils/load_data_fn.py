import random
import re
from enum import Enum, auto
from itertools import chain, islice
from math import ceil
from pathlib import Path

import pandas as pd
from toolkit.enums import Split
from toolkit.nlp import NLPTrainingConfig
from toolkit.nlp.data import ClassificationLabel, FinelyControlledText, PairedText, RegressionLabel
from transformers import PreTrainedTokenizer


class TextType(Enum):
    ORI = auto()


def load_data_fn4generate(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
    config = kwargs["config_load_data"]
    text_type = TextType[config.text_type]

    # t2s_ratio = re.search(r"mix_t2s_ratio=([\d\.]+)", model_name)
    # if t2s_ratio:
    #     t2s_ratio = float(t2s_ratio.group(1))
    # no_error_ratio = re.search(r"mix_no_error_ratio=([\d\.]+)", model_name)
    # if no_error_ratio:
    #     no_error_ratio = float(no_error_ratio.group(1))

    if isinstance(data_file_path, str | Path):
        data_file_path = Path(data_file_path)
        if data_file_path.is_dir():
            dfs = [pd.read_json(p) for p in data_file_path.iterdir()]
            iterator = chain(*[pd.read_json(p).iterrows() for p in data_file_path.iterdir()])
            n = sum((len(df) for df in dfs))
        else:
            df = pd.read_json(data_file_path, lines=False)
            iterator = df.iterrows()
            n = len(df)
    else:
        n = len(data_file_path)
        iterator = data_file_path.iterrows()

    inputs = []
    labels = []
    print(df.columns)
    for idx, row in iterator:
        # input
        match text_type:
            case TextType.ORI:
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": f'"{row["inputs"]}"下一句是?'},
                ]
                input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, continue_final_message=False)
                a_sample = PairedText(input_str)
        # label
        match text_type:
            case TextType.ORI:
                label_str = row["outputs"]
                if split == Split.TRAINING:
                    a_label = PairedText(label_str + EOS)
                else:
                    a_label = label_str
        inputs.append(a_sample)
        labels.append(a_label)

    print("=" * 30 + split.name + "=" * 30)
    print("### Input: ")
    print(inputs[0])
    print("### Output: ")
    print(labels[0])
    print("=" * 60)
    return inputs, labels


# def load_data_fn4generate(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
#     special_tokens_map = tokenizer.special_tokens_map
#     BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
#     EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
#     SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
#     MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
#     CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
#     config = kwargs["config_load_data"]
#     text_type = TextType[config.text_type]

#     # t2s_ratio = re.search(r"mix_t2s_ratio=([\d\.]+)", model_name)
#     # if t2s_ratio:
#     #     t2s_ratio = float(t2s_ratio.group(1))
#     # no_error_ratio = re.search(r"mix_no_error_ratio=([\d\.]+)", model_name)
#     # if no_error_ratio:
#     #     no_error_ratio = float(no_error_ratio.group(1))

#     if isinstance(data_file_path, str | Path):
#         data_file_path = Path(data_file_path)
#         if data_file_path.is_dir():
#             dfs = [pd.read_json(p) for p in data_file_path.iterdir()]
#             iterator = chain(*[pd.read_json(p).iterrows() for p in data_file_path.iterdir()])
#             n = sum((len(df) for df in dfs))
#         else:
#             df = pd.read_json(data_file_path, lines=False)
#             iterator = df.iterrows()
#             n = len(df)
#     else:
#         n = len(data_file_path)
#         iterator = data_file_path.iterrows()

#     inputs = []
#     labels = []
#     print(df.columns)
#     for idx, row in iterator:
#         # input
#         match text_type:
#             case TextType.ORI:
#                 messages = [
#                     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#                     {"role": "user", "content": f'"{row["inputs"]}"下一句是?'},
#                 ]
#                 input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, continue_final_message=False)
#                 a_sample = PairedText(input_str)
#         # label
#         match text_type:
#             case TextType.ORI:
#                 label_str = row["outputs"]
#                 a_label = PairedText(label_str + EOS)
#         inputs.append(a_sample)
#         labels.append(a_label)

#     print("=" * 30 + split.name + "=" * 30)
#     print("### Input: ")
#     print(inputs[0])
#     print("### Output: ")
#     print(labels[0])
#     print("=" * 60)
#     return inputs, labels


def load_data_fn4classify(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
    config = kwargs["config_load_data"]
    text_type = TextType[config.text_type]

    # t2s_ratio = re.search(r"mix_t2s_ratio=([\d\.]+)", model_name)
    # if t2s_ratio:
    #     t2s_ratio = float(t2s_ratio.group(1))
    # no_error_ratio = re.search(r"mix_no_error_ratio=([\d\.]+)", model_name)
    # if no_error_ratio:
    #     no_error_ratio = float(no_error_ratio.group(1))

    if isinstance(data_file_path, str | Path):
        data_file_path = Path(data_file_path)
        if data_file_path.is_dir():
            dfs = [pd.read_json(p) for p in data_file_path.iterdir()]
            iterator = chain(*[pd.read_json(p).iterrows() for p in data_file_path.iterdir()])
            n = sum((len(df) for df in dfs))
        else:
            df = pd.read_json(data_file_path, lines=False)
            iterator = df.iterrows()
            n = len(df)
    else:
        n = len(data_file_path)
        iterator = data_file_path.iterrows()

    inputs = []
    labels = []
    print(df.columns)
    for idx, row in iterator:
        # input
        match text_type:
            case TextType.ORI:
                input_str = row["inputs"]
                a_sample = PairedText(input_str)
        # label
        match text_type:
            case TextType.ORI:
                label_str = row["outputs"]
                a_label = ClassificationLabel(label_str)

        inputs.append(a_sample)
        labels.append(a_label)

    print("=" * 30 + split.name + "=" * 30)
    print("### Input: ")
    print(inputs[0])
    print("### Output: ")
    print(labels[0])
    print("=" * 60)
    return inputs, labels
