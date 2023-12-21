import torch
import transformers
import numpy as np
import json
import argparse
import math
import os

from tqdm.auto import tqdm
from accelerate import Accelerator
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    SchedulerType,
    DefaultDataCollator,
    DataCollatorForLanguageModeling
)
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers.models.auto.auto_factory import model_type_to_module_name
from torch.utils.data.dataloader import default_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="",
        help="Path to train data."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        help="Path to test data."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=300,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=80,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_len",
        default=512,
        type=int,
        help="Max length of inputs"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to output file."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--output_dir", type=str, default="HW3", help="Where to store the final model.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        "Helsinki-NLP/opus-mt-zh-vi"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-zh-vi",
        padding_side='left'
    )
    model.to("cuda")
    
    with open(args.train_data_path, "r") as f:
        train_data = json.load(f)
    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)
    
    data_files = {}
    data_files["train"] = args.train_data_path
    data_files["test"] = args.test_data_path
    raw_datasets = load_dataset("json", data_files=data_files)

    padding = "max_length"
    def preprocess_function_train(examples):
        inputs = examples["input"]
        targets = examples["output"]
        all_inputs = [inputs[i]+targets[i] for i in range(len(inputs))]
        tokenized_instructions = tokenizer(all_inputs, add_special_tokens=False)
        model_inputs = tokenized_instructions
        return model_inputs
    
    column_names = raw_datasets["train"].column_names

    train_dataset = raw_datasets["train"].map(
        preprocess_function_train,
        batched=True,
        remove_columns=column_names
    )

    def preprocess_function_test(examples):
        inputs = examples["input"]
        tokenized_instructions = tokenizer(inputs, add_special_tokens=False)
        return tokenized_instructions

    test_dataset = raw_datasets["test"].map(
        preprocess_function_test,
        batched=True,
        remove_columns=column_names
    )

    label_pad_token_id = -100
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.optimization.Adafactor(optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    if args.push_to_hub:
        repo_name = Path(args.output_dir).absolute().name
        # Create repo and retrieve repo_id
        repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
        # Clone repo locally
        repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)
        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    
    model.train()
    ppls = []
    for step, batch in enumerate(train_dataloader):
        #print(batch)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            print(loss)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1

    
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
        
    if args.push_to_hub:
        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
    

    model.eval()
    gen_kwargs = {"max_length": args.max_len, "num_beams": 5}
    all_results = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            #print(batch)
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(decoded_preds)
            all_results.extend(decoded_preds)
    results = [{"id": raw_datasets["test"]["id"][i], "output": all_results[i]}
    for i in range(len(all_results))]
    with open(args.output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
