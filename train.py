import json
import re
import time
from pathlib import Path

import toolkit
import torch
import torch.distributed as dist
from fire import Fire
from toolkit import getLogger
from toolkit.enums import Split
from toolkit.nlp import NLPTrainingConfig, TextDataset
from toolkit.training import Trainer, initialize
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizer

from utils.evaluators import Evaluator1
from utils.load_data_fn import load_data_fn

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_PAD_TOKEN = "<unk>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def load_tokenizer() -> PreTrainedTokenizer:
    # * Load tokenizer
    tokenizer_kwargs = {}
    if config.model_dir:
        tokenizer = AutoTokenizer.from_pretrained(config.model_dir, **tokenizer_kwargs, trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # * resize embedding
    if tokenizer.pad_token is None:
        logger.debug(f"Adding pad token {DEFAULT_PAD_TOKEN}")
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    logger.debug(f"len(tokenizer):{len(tokenizer)}")
    if dist.is_initialized():
        dist.barrier()
    return tokenizer


def load_dataset(tokenizer: PreTrainedTokenizer) -> tuple:
    # * Load training data, development data and test data
    # path = Path(config.train_file_path)
    # if path.is_dir():
    #     files = [os.path.join(path, file.name) for file in path.glob("*.json")]
    # else:
    #     files = str(path)
    # logger.debug(str(files))
    train_dataset = TextDataset.from_file(
        config.train_file_path,
        tokenizer,
        split=Split.TRAINING,
        configs=config,
        load_data_fn=load_data_fn,
        dataset=config.dataset_name,
        text_type=config.text_type,
        model_name=config.model_name,
    )
    val_dataset = TextDataset.from_file(
        config.val_file_path if "for_refiner" not in config.model_name else config.train_file_path,
        tokenizer,
        split=Split.VALIDATION,
        configs=config,
        load_data_fn=load_data_fn,
        dataset=config.dataset_name,
        text_type=config.text_type,
        model_name=config.model_name,
    )
    test_dataset = TextDataset.from_file(
        config.test_file_path,
        tokenizer,
        split=Split.TEST,
        configs=config,
        load_data_fn=load_data_fn,
        # train_config=config,
        dataset=config.dataset_name,
        text_type=config.text_type,
        model_name=config.model_name,
    )
    if dist.is_initialized():
        dist.barrier()

    if "debug" in config.model_name:
        train_dataset.batch_model_input = train_dataset.batch_model_input[: config.train_batch_size]
        # val_dataset.batch_model_input = val_dataset.batch_model_input[:config.train_batch_size]
        print("train dataset length:", len(train_dataset))

    print(f"-------------------------------TRAINING-------------------------------")
    print(f"\n### input ###\n{train_dataset.texts_input[0][0]}")
    print(f"\n### label ###\n{train_dataset.texts_label[0][0]}\n")

    if val_dataset is not None:
        print(f"-------------------------------VALIDATION-------------------------------")
        print(f"\n### input ###\n{val_dataset.texts_input[0][0]}")
        print(f"\n### label ###\n{val_dataset.texts_label[0]}\n")

    if test_dataset is not None:
        print(f"-------------------------------TEST-------------------------------")
        print(f"\n### input ###\n{test_dataset.texts_input[0][0]}")
        print(f"\n### label ###\n{test_dataset.texts_label[0]}\n")

    return train_dataset, val_dataset, test_dataset


def load_hf_generation_config(model, config):
    if hasattr(config, "hf_gen_config_file"):
        with open(config.hf_gen_config_file, "r") as f:
            data = f.read()
            d = json.loads(data)
            print(d)
            # hf_generate_config = GenerationConfig.from_dict(d)
            if model is not None:
                from copy import deepcopy

                hf_generate_config = deepcopy(model.generation_config)
            else:
                hf_generate_config = GenerationConfig.from_pretrained(config.model_dir)
            hf_generate_config.update(**d)
            print("\nhf generation config:")
            print(hf_generate_config)
            return hf_generate_config


def load_model(tokenizer):
    # global dschf
    start = time.time()

    # * define model class
    model_class = AutoModelForCausalLM

    # * define from_pretrained kwargs
    from_pretrained_kwargs = None

    # * Load model config
    # model_kwargs = {"cache_dir": config.cache_dir}  # "revision": config.model_revision, "use_auth_token": True if config.use_auth_token else None
    model_kwargs = {}
    if config.model_dir:
        model_config = AutoConfig.from_pretrained(config.model_dir, **model_kwargs, trust_remote_code=True)
    else:
        model_config = CONFIG_MAPPING[config.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if config.config_overrides is not None:
            logger.info(f"Overriding config: {config.config_overrides}")
            model_config.update_from_string(config.config_overrides)
            logger.info(f"New config: {config}")

    # * Load model
    if config.model_dir:
        if config.parallel_mode != "deepspeed":
            torch_dtype = config.torch_dtype if config.torch_dtype in ["auto", None] else getattr(torch, config.torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                config=model_config,
                cache_dir=config.cache_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        else:
            logger.debug(f"local_rank {local_rank}: Construct `from_pretrained` kwargs ...")
            model = None
            torch_dtype = config.torch_dtype if config.torch_dtype in ["auto", None] else getattr(torch, config.torch_dtype)
            from_pretrained_kwargs = dict(cache_dir=config.cache_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=False, trust_remote_code=True)
    else:
        exit(1)
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.debug(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    if model_config.vocab_size != len(tokenizer):
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != embedding_size:
            logger.debug("resize the embedding size by the size of the tokenizer")
            model.resize_token_embeddings(len(tokenizer))

    end = time.time()
    logger.debug(f"local_rank {local_rank}: Loading model takes {end - start:.2f} sec.")

    # * sync
    if dist.is_initialized():
        dist.barrier()
    return model, model_config, model_class, from_pretrained_kwargs


def main() -> None:
    # * Loading tokenizer
    tokenizer = load_tokenizer()

    # * load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(tokenizer)

    # *load model
    model, model_config, model_class, from_pretrained_kwargs = load_model(tokenizer)

    # *load generation config
    hf_gen_config = load_hf_generation_config(model, config)

    # * Train
    trainer = Trainer(
        task_type=config.task_type,
        evaluate_only=False,
        config=config,
        model=model,
        model_config=model_config,
        model_class=model_class,
        from_pretrained_kwargs=from_pretrained_kwargs,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        tokenizer=tokenizer,
        extral_evaluators=[Evaluator1],
        optimizer=config.opt_type,
        scheduler=config.sch_type if config.parallel_mode is not None else "linearWarmupDecay",
        extral_args_evaluation={"generation_config": hf_gen_config},
    )
    trainer.train()


if __name__ == "__main__":
    # * Get args
    config: NLPTrainingConfig = Fire(NLPTrainingConfig, silence=True)

    # * Create checkpoints,tensorboard outputs directory
    _dir = Path(
        config.dataset_name,
        config.model_type,
        config.text_type,
        config.model_name,
        str(config.epochs),
        str(config.train_batch_size),
        str(config.opt_lr),
        str(config.seed),
    )
    if config.save_dir is None:
        config.save_dir = Path("outputs", _dir)
    config.save(config.save_dir, silence=False)

    # * Create logger
    output_path_logger = config.save_dir / "report.log"
    logger = getLogger(__name__, output_path_logger)
    toolkit.set_file_logger(output_path_logger)

    # * Initalize parallel and seed
    local_rank, world_size = initialize(config)

    main()
