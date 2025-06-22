from toolkit.enums import Split
from toolkit.nlp.data import ClassificationLabel, FinelyControlledText, PairedText, RegressionLabel
from transformers import PreTrainedTokenizer

from utils.load_data_fn import TextType


def parse_item_fn4classify(item, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
    config = kwargs["config_load_data"]
    text_type = TextType[config.text_type]

    inputs, label = item.split("\t")
    # input
    match text_type:
        case TextType.ORI:
            input_str = inputs
            a_sample = PairedText(input_str)
    # label
    match text_type:
        case TextType.ORI:
            label_str = int(label)
            a_label = ClassificationLabel(label_str)

    return a_sample, a_label


def parse_item_fn4generate(item, tokenizer: PreTrainedTokenizer, split: Split, **kwargs):
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

    inputs, label = item.split("\t")
    # input
    match text_type:
        case TextType.ORI:
            if tokenizer.chat_template is not None:
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},  # 也许 system 会不同
                    {"role": "user", "content": f'"{inputs}"下一句是?'},
                ]
                input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, continue_final_message=False)
            else:
                input_str = f'"{inputs}"下一句是?'
                # input_str = f'What is the next sentence of "{translation_dict[inputs]}"?'
            a_sample = PairedText(input_str)
    # label
    match text_type:
        case TextType.ORI:
            label_str = label
            # label_str = translation_dict[label]
            if model_structure == "decoder":
                # 必须手动处理 EOS, 因为对 encoder 模型的 labels 进行 tokenize 时, 设置了 add_special_tokens=False
                a_label = PairedText(label_str + EOS)
            elif model_structure == "encoder-decoder":
                # 好像一般 encoder-decoder 模型的 tokenizer 会自动添加 EOS, 至少 Flan-t5 的 tokenizer 会自动加
                a_label = PairedText(label_str)

    # print("=" * 30 + split.name + "=" * 30)
    # print("### Input: ")
    # print(inputs)
    # print("### Output: ")
    # print(label)
    # print("=" * 60)
    return a_sample, a_label
