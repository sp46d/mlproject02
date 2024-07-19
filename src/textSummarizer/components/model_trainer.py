import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = self.config
        
        if not os.path.exists(os.path.join(config.root_dir, config.out_dir)):
        
            if self.config.use_fine_tuned_model:
                model = AutoModelForSeq2SeqLM.from_pretrained(config.fine_tuned_model)
                tokenizer = AutoTokenizer.from_pretrained(config.fine_tuned_model)
                logger.info(f"Fine-tuned model {config.fine_tuned_model} has been downloaded directly from the HuggingFace hub")

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

                logger.info(f"Fine-tuning {config.base_model} on data at {config.data_path} is about to begin")
                trainer.train()
                logger.info(f"Fine-tuning {config.base_model} on data at {config.data_path} has been successfully completed")
                
                trainer.model.save_pretrained("final_checkpoint")
                
                base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
                peft_model = PeftModel(base_model, "final_checkpoint")
                model = peft_model.merge_and_unload()
                logger.info("The trained Lora adapter has been merged to the base model")
                
            # Save model
            model.save_pretrained(os.path.join(config.root_dir, config.out_dir))
            # Save tokenizer
            tokenizer.save_pretrained(os.path.join(config.root_dir, "tokenizer"))
            
        else:
            logger.info(f"Model {config.out_dir} already exsits, so the model training part has been skipped.")