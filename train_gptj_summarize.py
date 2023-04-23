import random

import evaluate
import numpy as np
import torch
from summarize_dataset import TLDRDataset

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    MegaForCausalLM
)

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

if __name__ == "__main__":
    output_dir = "/content/drive/MyDrive/gptj"
    train_batch_size = 16
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    eval_batch_size = 5
    eval_steps = 1000
    max_input_length = 1024
    save_steps = 1000
    num_train_epochs = 10
    random.seed(42)

    config = AutoConfig.from_pretrained("mnaylor/mega-base-wikitext")
    config.is_decoder = True
    config.bidirectional = False
    config.vocab_size = 7700
    config.pad_token_id = 0
    config.bos_token_id = 0
    config.eos_token_id = 0
    config.max_positi=ons=1024
    config.num_hidden_layers=16
    config.hidden_size=1024
    config.nffn_hidden_size=1536
    config.intermediate_size = 1536
    config.ema_projection_size = 16
    config.shared_representation_size = 256

    model = MegaForCausalLM.from_pretrained("/content/drive/MyDrive/gptj")
    model = model.cuda()

    # Set up the datasets
    data_path = "CarperAI/openai_summarize_tldr"
    train_dataset = TLDRDataset(
        "train",
        max_length=1024,
    )
    dev_dataset = TLDRDataset(
        "valid",
        max_length=1024,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        half_precision_backend=True,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=50,
        deepspeed="./ds_config_gptj.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)

    print(model.generate(inputs=torch.tensor([[2]]).long().cuda(),do_sample=True, max_length=50, top_k=50))
