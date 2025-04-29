import random
import re
from enum import Enum, auto
from itertools import chain, islice
from math import ceil
from pathlib import Path

import pandas as pd
from toolkit.enums import Split
from toolkit.nlp import NLPTrainingConfig
from toolkit.nlp.data import (
    ClassificationLabel,
    FinelyControlledText,
    PairedText,
    RegressionLabel,
)
from transformers import PreTrainedTokenizer


class TextType(Enum):
    ORI = auto()


def load_data_fn4generate(data_file_path: Path | str, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
    config = kwargs["config_load_data"]
    text_type = TextType[config.text_type]
    model_type = config.model_type
    model_structure = config.model_structure

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
                if tokenizer.chat_template is not None:
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},  # 也许 system 会不同
                        {"role": "user", "content": f'"{row["inputs"]}"下一句是?'},
                    ]
                    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, continue_final_message=False)
                else:
                    input_str = f'"{row["inputs"]}"下一句是?'
                a_sample = PairedText(input_str)
        # label
        match text_type:
            case TextType.ORI:
                label_str = row["outputs"]
                if split == Split.TRAINING:
                    if model_structure == "decoder":
                        # 必须手动处理 EOS, 因为对 encoder 模型的 labels 进行 tokenize 时, 设置了 add_special_tokens=False
                        a_label = PairedText(label_str + EOS)
                    elif model_structure == "encoder-decoder":
                        # 好像一般 encoder-decoder 模型的 tokenizer 会自动添加 EOS, 至少 Flan-t5 的 tokenizer 会自动加
                        a_label = PairedText(label_str)
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


def load_data_fn4classify(data_file_path: Path | str, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
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


def load_data_fn4generate_hf(data_file_path: Path | str, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
    """
    HF trainer 要求 labels 只能是 tensor
    """
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
    config = kwargs["config_load_data"]
    text_type = TextType[config.text_type]
    model_type = config.model_type
    model_structure = config.model_structure

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
                if tokenizer.chat_template is not None:
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},  # 也许 system 会不同
                        {"role": "user", "content": f'"{row["inputs"]}"下一句是?'},
                    ]
                    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, continue_final_message=False)
                else:
                    input_str = f'"{row["inputs"]}"下一句是?'
                    # input_str = f'What is the next sentence of "{translation_dict[row["inputs"]]}"?'
                a_sample = PairedText(input_str)
        # label
        match text_type:
            case TextType.ORI:
                label_str = row["outputs"]
                # label_str = translation_dict[row["outputs"]]
                if model_structure == "decoder":
                    # 必须手动处理 EOS, 因为对 encoder 模型的 labels 进行 tokenize 时, 设置了 add_special_tokens=False
                    a_label = PairedText(label_str + EOS)
                elif model_structure == "encoder-decoder":
                    # 好像一般 encoder-decoder 模型的 tokenizer 会自动添加 EOS, 至少 Flan-t5 的 tokenizer 会自动加
                    a_label = PairedText(label_str)
        inputs.append(a_sample)
        labels.append(a_label)

    print("=" * 30 + split.name + "=" * 30)
    print("### Input: ")
    print(inputs[0])
    print("### Output: ")
    print(labels[0])
    print("=" * 60)
    return inputs, labels


# translation_dict = {
#     "赵客缦胡缨": "The warriors of Zhao wear loose barbarian tassels",
#     "吴钩霜雪明": "Their swords shine like frost and snow",
#     "银鞍照白马": "Silver saddles gleam on white horses",
#     "飒沓如流星": "They race like meteors across the sky",
#     "十步杀一人": "In ten steps they slay one man",
#     "千里不留行": "And travel a thousand miles without a trace",
#     "事了拂衣去": "When their task is done they brush off their sleeves and leave",
#     "深藏身与名": "Hiding both their identity and their name",
#     "闲过信陵饮": "Leisurely they drink at Lord Xinlings feast",
#     "脱剑膝前横": "Their swords laid across their knees",
#     "将炙啖朱亥": "They offer roasted meat to Zhu Hai",
#     "持觞劝侯嬴": "And raise their cups to urge Hou Ying",
#     "三杯吐然诺": "After three cups they make solemn promises",
#     "五岳倒为轻": "Even the Five Sacred Peaks seem light in comparison",
#     "眼花耳热后": "With blurred vision and heated ears",
#     "意气素霓生": "Their spirit rises like a rainbow",
#     "救赵挥金槌": "To save Zhao they wield golden hammers",
#     "邯郸先震惊": "And Handan trembles at their might",
#     "千秋二壮士": "For a thousand autumns two heroic men",
#     "烜赫大梁城": "Have been illustrious in Daliang City",
#     "纵死侠骨香": "Even in death their heroic bones exude fragrance",
#     "不惭世上英": "And they are not ashamed to be called heroes of the world",
#     "谁能书阁下": "Who can write in the pavilion below",
#     "白首太玄经": "The profound classic until old age",
# }
