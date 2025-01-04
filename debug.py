import pandas as pd
from transformers import AutoTokenizer
import logging
from datasets import Dataset

def configurar_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analisar_tokens_dataset():
    logger = configurar_logging()
    logger.info("Iniciando análise de tokens do dataset...")

    try:
        # Carrega o dataset
        df = pd.read_json('./formatted_training_data.json')
        dataset = Dataset.from_pandas(df)
        logger.info(f"Dataset carregado com sucesso. Total de exemplos: {len(dataset)}")

        dataset = Dataset.from_list(dataset["train"])

        # Carrega o tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            trust_remote_code=True,
            cache_dir="./models"
        )
        logger.info("Tokenizer carregado com sucesso")

        # Analisa cada exemplo
        max_tokens = 0
        max_question = ""
        token_lengths = []

        for idx, row in enumerate(dataset):
            question = f"#### Human: {row['question'].strip()}"
            
            # Tokeniza a entrada completa
            tokens = tokenizer(question)
            num_tokens = len(tokens['input_ids'])
            token_lengths.append(num_tokens)
            
            if num_tokens > max_tokens:
                max_tokens = num_tokens
                max_question = question
        # Estatísticas
        logger.info(f"Número máximo de tokens: {max_tokens}")
        logger.info(f"Média de tokens: {sum(token_lengths) / len(token_lengths):.2f}")
        logger.info(f"Exemplo com mais tokens:\n{max_question[:500]}...")  # Mostra apenas os primeiros 500 caracteres
        
        return {
            "max_tokens": max_tokens,
            "max_question": max_question,
            "media_tokens": sum(token_lengths) / len(token_lengths),
            "distribuicao_tokens": token_lengths
        }

    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}")
        raise

if __name__ == "__main__":
    resultados = analisar_tokens_dataset()
