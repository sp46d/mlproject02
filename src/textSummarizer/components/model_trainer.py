from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from textSummarizer.entity import ModelTrainerConfig
import torch


# For now, we are not training the model, but load the model we want from HuggingFace hub
# and use it for inference

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = self.config
        
        if self.config.use_fine_tuned_model:
            model = AutoModelForSeq2SeqLM.from_pretrained(config.fine_tuned_model)
            tokenizer = AutoTokenizer.from_pretrained(config.fine_tuned_model)

        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model).to(device)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            
            #loading data
            dataset = load_from_disk(config.data_path)
            
            # set configs for training
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
            
            sft_config = SFTConfig(
                dataset_text_field=config.dataset_text_field,
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=config.per_device_eval_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                logging_steps=config.logging_steps,
                max_seq_length=config.max_seq_length,
                output_dir=config.root_dir
            )
        
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                tokenizer=tokenizer,
                peft_config=peft_config,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test']
            )
        
            trainer.train()
        
        # Save model
        model.save_pretrained(os.path.join(config.root_dir, "pegasus-xsum-6k"))
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(config.root_dir, "tokenizer"))
        