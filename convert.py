import json

def converter_formato(arquivo_entrada, arquivo_saida):
    # Estrutura do resultado final
    resultado = {
        "train": []
    }
    
    # Lê o arquivo de entrada linha por linha
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            try:
                # Carrega o objeto JSON da linha
                item = json.loads(linha)
                messages = item['messages']
                
                # Extrai as mensagens relevantes
                user_msg = messages[1]['content']
                assistant_msg = messages[2]['content']
                
                # Cria o novo formato
                novo_item = {
                    "question": user_msg.strip(),
                    "answer": assistant_msg.strip()
                }
                
                # Adiciona ao array 'train'
                resultado["train"].append(novo_item)
                
            except Exception as e:
                print(f"Erro ao processar linha: {e}")
                continue
    
    # Escreve o arquivo JSON final
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)

# Uso
converter_formato('training_data.jsonl', 'formatted_training_data.json')