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

max_length = 128


# Model loading params
load_in_4bit = True

# LoRA Params
lora_alpha = 16             # How much to weigh LoRA params over pretrained params
lora_dropout = 0.1          # Dropout for LoRA weights to avoid overfitting
lora_r = 16                 # Bottleneck size between A and B matrix for LoRA params
lora_bias = "all"           # "all" or "none" for LoRA bias
model_type = "llama"        # falcon or llama or wizard7 or wizard13
dataset_type = "custom"     # "squad" or "reddit" or "reddit_negative" or "custom"
lora_target_modules = [     # Which modules to apply LoRA to (names of the modules in state_dict)
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
] if model_type == "falcon" else [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# Trainer params
output_dir = "outputs_custom"                            # Directory to save the model
optim_type = "adafactor"                            # Optimizer type to train with 
learning_rate = 0.00005                              # Model learning rate
weight_decay = 0.002                                # Model weight decay
per_device_train_batch_size = 8                     # Train batch size on each GPU
per_device_eval_batch_size = 8                      # Eval batch size on each GPU
gradient_accumulation_steps = 2                     # Number of steps before updating model
warmup_steps = 5                                    # Number of warmup steps for learning rate
save_steps = 100                                     # Number of steps before saving model
logging_steps = 25                                  # Number of steps before logging





# Load in the model as a 4-bit or 8-bit model
if load_in_4bit == True:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
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



# Load in the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "WizardLM/WizardLM-13B-V1.2" if model_type == "wizard13" \
            else "TheBloke/wizardLM-7B-HF" if model_type == "wizard7" \
            else "tiiuae/falcon-7b" if model_type == "falcon" \
            else "meta-llama/Llama-3.3-70B-Instruct",
    trust_remote_code=True,
    cache_dir="./models",
)
tokenizer.pad_token = tokenizer.eos_token


# Carregando com pandas primeiro
df = pd.read_json('./formatted_training_data.json')
dataset = Dataset.from_pandas(df)

def map_function(example):
    # Get the question and model output
    question = f"#### Human: {example['question'].strip()}"
    output = f"#### Assistant: {example['answer'].strip()}"

    # Encode the question and output
    question_encoded = tokenizer(question)
    output_encoded = tokenizer(output, max_length=max_length-1-len(question_encoded["input_ids"]), truncation=True, padding="max_length")

    # Add on a pad token to the end of the input_ids
    output_encoded["input_ids"] = output_encoded["input_ids"] + [tokenizer.pad_token_id]
    output_encoded["attention_mask"] = output_encoded["attention_mask"] + [0]

    # Combine the input ids
    input_ids = question_encoded["input_ids"] + output_encoded["input_ids"]

    # Combine the attention masks. Attention masks are 0
    # where we want to mask and 1 where we want to attend.
    # We want to attend to both context and generated output
    # Also add a 1 for a single padding
    attention_mask = [1]*len(question_encoded["input_ids"]) + [1]*(sum(output_encoded["attention_mask"])+1) + [0]*(len(output_encoded["attention_mask"])-sum(output_encoded["attention_mask"])-1)
    
    # The labels are the input ids, but we want to mask the loss for the context and padding
    labels = [input_ids[i] if attention_mask[i] == 1 else -100 for i in range(len(attention_mask))]
    assert len(labels) == len(attention_mask) and len(attention_mask) == len(input_ids), "Labels is not the correct length"

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

dataset = Dataset.from_list(dataset["train"]).map(map_function)

Randomize data
dataset = dataset.shuffle()

# Test/train split
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
data_train = dataset.select(range(train_size))
data_test = dataset.select(range(train_size, train_size + test_size))


# Adapt the model with LoRA weights
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias=lora_bias,
    task_type="CAUSAL_LM",
    inference_mode=False,
    target_modules=lora_target_modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    optim=optim_type,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    do_train=True,
    warmup_steps=warmup_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_test,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
