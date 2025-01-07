from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import pandas as pd
from datasets import Dataset
import logging
import sys
import torch
import gc

torch.cuda.empty_cache()

def configurar_logging():
    class TruncatingFormatter(logging.Formatter):
        def format(self, record):
            # Formata a mensagem normalmente
            message = super().format(record)
            # Trunca para 100 caracteres
            return message[:100] + '...' if len(message) > 100 else message

    formatter = TruncatingFormatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configura os handlers com o novo formatter
    file_handler = logging.FileHandler('finetune.log')
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # Configura o logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, stream_handler]
    )
    return logging.getLogger(__name__)

def carregar_modelo(model_type, load_in_4bit, logger):
    try:
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_type="float16"
            )
            model = AutoModelForCausalLM.from_pretrained(
                "WizardLM/WizardLM-13B-V1.2" if model_type == "wizard13" \
                    else "TheBloke/wizardLM-7B-HF" if model_type == "wizard7" \
                    else "tiiuae/falcon-7b" if model_type == "falcon" \
                    else "meta-llama/Llama-3.3-70B-Instruct",
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb_config,
                cache_dir="./models",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "WizardLM/WizardLM-13B-V1.2" if model_type == "wizard13" \
                    else "TheBloke/wizardLM-7B-HF" if model_type == "wizard7" \
                    else "tiiuae/falcon-7b" if model_type == "falcon" \
                    else "meta-llama/Llama-3.3-70B-Instruct",
                trust_remote_code=True,
                device_map="auto",
                load_in_8bit=True,
                cache_dir="./models",
            )
        logger.info("Modelo carregado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {str(e)}")
        raise

def carregar_tokenizer(model_type, logger):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "WizardLM/WizardLM-13B-V1.2" if model_type == "wizard13" \
                    else "TheBloke/wizardLM-7B-HF" if model_type == "wizard7" \
                    else "tiiuae/falcon-7b" if model_type == "falcon" \
                    else "meta-llama/Llama-3.3-70B-Instruct",
            trust_remote_code=True,
            cache_dir="./models",
        )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer carregado com sucesso")
        return tokenizer
    except Exception as e:
        logger.error(f"Erro ao carregar o tokenizer: {str(e)}")
        raise

def carregar_dataset(logger):
    try:
        df = pd.read_json('./formatted_training_data.json')
        dataset = Dataset.from_pandas(df)
        logger.info(f"Dataset carregado com sucesso. Tamanho: {len(dataset)}")
        return dataset
    except Exception as e:
        logger.error(f"Erro ao carregar o dataset: {str(e)}")
        raise

def processar_dataset(dataset, tokenizer, max_length, logger):
    def map_function(example):
        try:
            logger.info("Iniciando processamento de exemplo...")
            
            question = f"#### Human: {example['question'].strip()}"
            output = f"#### Assistant: {example['answer'].strip()}"
            logger.debug(f"Questão formatada: {question}")
            logger.debug(f"Resposta formatada: {output}")

            question_encoded = tokenizer(question)
            logger.debug(f"Tamanho dos tokens da questão: {len(question_encoded['input_ids'])}")
            
            max_length_output = max(1, max_length-1-len(question_encoded["input_ids"]))
            output_encoded = tokenizer(output, max_length=max_length_output, truncation=True, padding="max_length")
            logger.debug(f"Tamanho dos tokens da resposta: {len(output_encoded['input_ids'])}")

            output_encoded["input_ids"] = output_encoded["input_ids"] + [tokenizer.pad_token_id]
            output_encoded["attention_mask"] = output_encoded["attention_mask"] + [0]

            input_ids = question_encoded["input_ids"] + output_encoded["input_ids"]
            logger.debug(f"Tamanho total dos input_ids: {len(input_ids)}")

            attention_mask = [1]*len(question_encoded["input_ids"]) + [1]*(sum(output_encoded["attention_mask"])+1) + [0]*(len(output_encoded["attention_mask"])-sum(output_encoded["attention_mask"])-1)
            logger.debug(f"Tamanho da attention mask: {len(attention_mask)}")
            
            labels = [input_ids[i] if attention_mask[i] == 1 else -100 for i in range(len(attention_mask))]
            assert len(labels) == len(attention_mask) and len(attention_mask) == len(input_ids), "Labels is not the correct length"

            logger.info("Exemplo processado com sucesso")
            result = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
            gc.collect()
            return result
        except Exception as e:
            logger.error(f"Erro no processamento do exemplo: {str(e)}")
            raise

    try:
        dataset = Dataset.from_list(dataset["train"]).map(map_function)
        torch.cuda.empty_cache()
        logger.info("Transformações aplicadas com sucesso")
        return dataset
    except Exception as e:
        logger.error(f"Erro ao aplicar transformações no dataset: {str(e)}")
        raise

def preparar_dados_treino(dataset, logger):
    dataset = dataset.shuffle()
    logger.info("Dataset embaralhado")

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    data_train = dataset.select(range(train_size))
    data_test = dataset.select(range(train_size, train_size + test_size))
    logger.info(f"Split realizado - Train size: {train_size}, Test size: {test_size}")
    
    return data_train, data_test

def configurar_lora(model, lora_config, logger):
    try:
        logger.info(f"Configurando LoRA: {lora_config}")
        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info("LoRA configurado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao configurar LoRA: {str(e)}")
        raise

def configurar_trainer(model, training_args_config, data_train, data_test, tokenizer, logger):
    try:
        logger.info(f"Configurando Trainer: {training_args_config}")
        training_args = TrainingArguments(**training_args_config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_test,
            tokenizer=tokenizer,
        )
        logger.info("Trainer configurado com sucesso")
        return trainer
    except Exception as e:
        logger.error(f"Erro ao configurar trainer: {str(e)}")
        raise

def treinar_modelo(trainer, logger):
    try:
        logger.info("Iniciando treinamento...")
        trainer.train()
        logger.info("Treinamento concluído com sucesso")
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        raise

def main():
    # Configurações
    max_length = 512
    load_in_4bit = True
    model_type = "wizard7"
    
    lora_config = {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "r": 16,
        "bias": "all",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    }
    
    training_args_config = {
        "output_dir": "outputs_custom",
        "evaluation_strategy": "epoch",
        "optim": "adafactor",
        "learning_rate": 0.00005,
        "weight_decay": 0.002,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "do_train": True,
        "warmup_steps": 5,
        "save_steps": 100,
        "logging_steps": 25,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False
        },
        "max_grad_norm": 0.3,
        "fp16": True,
    }

    # Inicialização
    logger = configurar_logging()
    logger.info("Iniciando processo de fine-tuning...")
    
    # Pipeline principal
    model = carregar_modelo(model_type, load_in_4bit, logger)
    tokenizer = carregar_tokenizer(model_type, logger)
    dataset = carregar_dataset(logger)
    dataset = processar_dataset(dataset, tokenizer, max_length, logger)
    data_train, data_test = preparar_dados_treino(dataset, logger)
    model = configurar_lora(model, lora_config, logger)
    trainer = configurar_trainer(model, training_args_config, data_train, data_test, tokenizer, logger)
    treinar_modelo(trainer, logger)

if __name__ == "__main__":
    main()
