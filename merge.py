import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
import os

# Configurações
lora_path = "outputs_custom/checkpoint-100"  # Path para os pesos do LoRA
output_path = "outputs_custom/merged_model"  # Path para salvar o modelo mesclado
model_name = "meta-llama/Llama-3.3-70B-Instruct"  # Nome do modelo base

# Carrega a configuração do LoRA
peft_config = PeftConfig.from_pretrained(lora_path)

# Carrega o modelo base
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu", 
    cache_dir="./models"
)

# Carrega o tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="./models"
)

# Copia os arquivos do modelo base para o diretório de saída
path = os.path.join("models", "models--meta-llama--Llama-3.3-70B-Instruct", "snapshots")
if os.path.exists(path):
    snapshot_folder = os.listdir(path)[0]
    source_path = os.path.join(path, snapshot_folder)
    shutil.copytree(source_path, output_path, dirs_exist_ok=True, 
                    ignore=shutil.ignore_patterns('*.pt', "*.pth", "*.bin"))

# Carrega e mescla o modelo LoRA
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# Mescla os adaptadores LoRA
key_list = [key for key, _ in model.named_modules() if "lora" not in key]
for key in key_list:
    try:
        sub_mod = model.get_submodule(key)
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
    except AttributeError:
        continue
        
    target_name = key.split(".")[-1]
    if isinstance(sub_mod, peft.tuners.lora.Linear):
        sub_mod.merge()
        bias = sub_mod.bias is not None
        new_module = torch.nn.Linear(sub_mod.in_features, sub_mod.out_features, bias=bias)
        new_module.weight.data = sub_mod.weight
        if bias:
            new_module.bias.data = sub_mod.bias
        model.base_model._replace_module(parent, target_name, new_module, sub_mod)

# Obtém o modelo base após a mesclagem
model = model.base_model.model

# Salva o modelo mesclado
print(f"Salvando modelo mesclado em {output_path}")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("Modelo mesclado salvo com sucesso!")