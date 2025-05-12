import argparse
import sys
import os
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - TEST - %(levelname)s - %(message)s')

# Adicionar o diretório atual ao PATH para importar os módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importar a função de teste do Self-RAG
try:
    from src.self_rag import run_self_rag_tests
except ImportError as e:
    print(f"Erro ao importar módulo Self-RAG: {e}")
    print("Certifique-se de que o arquivo src/self_rag.py existe e contém a função run_self_rag_tests.")
    sys.exit(1)

def main():
    """
    Executa testes automatizados do sistema Self-RAG.
    
    Este script permite testar o sistema Self-RAG com uma série de perguntas
    predefinidas relacionadas aos acórdãos 733/2025 e 764/2025, gerando
    um relatório detalhado dos resultados.
    """
    parser = argparse.ArgumentParser(description="Testes automatizados do sistema Self-RAG")
    
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Diretório para salvar os resultados")
    parser.add_argument("--initial_k", type=int, default=10,
                        help="Número de documentos iniciais a recuperar")
    parser.add_argument("--reranker_n", type=int, default=5,
                        help="Número de documentos após reranking")
    parser.add_argument("--max_attempts", type=int, default=3,
                        help="Máximo de tentativas de refinamento por pergunta")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperatura do LLM para geração de respostas")
    parser.add_argument("--max_tokens", type=int, default=250,
                        help="Máximo de tokens por resposta gerada")
    parser.add_argument("--select_category", type=str, default="",
                        help="Categoria específica para testar (acordao_733_2025, acordao_764_2025 ou questoes_complexas). Se vazio, testa todas.")
    parser.add_argument("--select_question", type=str, default="",
                        help="ID específico de uma pergunta para testar (ex: 733_simples_1). Se vazio, testa todas da categoria.")
    
    args = parser.parse_args()
    
    print("\n" + "="*30 + " TESTES AUTOMATIZADOS SELF-RAG " + "="*30)
    print("Configurações:")
    print(f"- Diretório de saída: {args.output_dir}")
    print(f"- Recuperação inicial (k): {args.initial_k}")
    print(f"- Documentos após reranking (n): {args.reranker_n}")
    print(f"- Máximo de tentativas: {args.max_attempts}")
    print(f"- Temperatura LLM: {args.temperature}")
    print(f"- Máximo de tokens por resposta: {args.max_tokens}")
    
    if args.select_category:
        print(f"- FILTRO: Apenas categoria '{args.select_category}'")
    if args.select_question:
        print(f"- FILTRO: Apenas pergunta ID '{args.select_question}'")
    
    # Executar os testes
    try:
        print("\nIniciando testes...")
        start_time = time.time()
        
        results = run_self_rag_tests(
            output_dir=args.output_dir,
            initial_k=args.initial_k,
            reranker_n=args.reranker_n,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            select_category=args.select_category,
            select_question=args.select_question
        )
        
        print(f"\nTestes concluídos em {time.time() - start_time:.2f} segundos.")
        print(f"Resultados completos salvos em: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Erro durante a execução dos testes: {e}")
        print(f"\nERRO: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 