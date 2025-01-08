from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configurações
device = "auto"
local_model_path = "outputs_custom/merged_model"  # Caminho para o modelo mesclado
repo_name = "tsrrodrigues/legal_doc_llama_v1"      # Nome do seu repositório no HuggingFace
hf_token = "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"                      # Seu token do HuggingFace

print("Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    device_map=device,
    torch_dtype=torch.float16,
    cache_dir="./models"
).eval()

print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    cache_dir="./models"
)

print(f"Enviando modelo para {repo_name}...")
model.push_to_hub(
    repo_name, 
    token=hf_token,
    private=True  # Defina como False se quiser que o repositório seja público
)

print("Enviando tokenizer...")
tokenizer.push_to_hub(
    repo_name, 
    token=hf_token,
    private=True  # Mantenha consistente com a configuração do modelo
)

print("Upload concluído com sucesso!")