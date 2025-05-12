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
    parser.add_argument("--debug", action="store_true",
                        help="Ativa modo de debug com mais informações de diagnóstico")
    
    args = parser.parse_args()
    
    # Se modo debug ativado, configurar logging para DEBUG
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Modo DEBUG ativado")
        
        # Verificar ambiente e configurações
        print("\n=== DIAGNÓSTICO DO AMBIENTE ===")
        print(f"Diretório atual: {os.getcwd()}")
        print(f"Path do Python: {sys.path}")
        print(f"Verificando diretório ChromaDB: {os.path.join(os.getcwd(), 'chroma_db_index')}")
        
        # Verificar se o diretório ChromaDB existe
        chroma_dir = os.path.join(os.getcwd(), 'chroma_db_index')
        if os.path.exists(chroma_dir):
            print(f"✓ Diretório ChromaDB existe em: {chroma_dir}")
            # Listar conteúdo do diretório
            print("Conteúdo do diretório ChromaDB:")
            for item in os.listdir(chroma_dir):
                print(f"  - {item}")
            
            # Verificar se existe collections.sqlite3
            if os.path.exists(os.path.join(chroma_dir, 'collections.sqlite3')):
                print("✓ Arquivo collections.sqlite3 encontrado")
                
                # Se possível, verificar collections no ChromaDB
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=chroma_dir)
                    collections = client.list_collections()
                    print(f"Collections no ChromaDB ({len(collections)}):")
                    for collection in collections:
                        print(f"  - {collection.name} ({collection.count()} itens)")
                except Exception as e:
                    print(f"Erro ao acessar ChromaDB: {e}")
        else:
            print(f"✗ Diretório ChromaDB NÃO existe em: {chroma_dir}")
        
        # Verificar imports
        print("\nVerificando imports:")
        required_modules = ["torch", "transformers", "sentence_transformers", "chromadb"]
        for module in required_modules:
            try:
                __import__(module)
                print(f"✓ Módulo {module} importado com sucesso")
            except ImportError as e:
                print(f"✗ Erro ao importar {module}: {e}")
                
        print("=== FIM DO DIAGNÓSTICO ===\n")
    
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
        
        # Gerar relatório final
        try:
            print(f"\n{'='*30} RELATÓRIO RESUMIDO {'='*30}")
            print(f"Total de perguntas: {results['summary']['total_questions']}")
            print(f"Tempo total de processamento: {results['summary']['processing_time']:.2f}s")
            
            # Evitar divisão por zero nos cálculos de média
            if results['summary']['total_questions'] > 0:
                print(f"Qualidade média das respostas: {results['summary']['avg_quality_score']:.2f}/10")
                print(f"Média de tentativas por pergunta: {results['summary']['avg_attempts']:.2f}/{args.max_attempts}")
                print(f"Total de tokens consumidos: {results['summary']['total_tokens']}")
                print(f"Tokens adicionais (avaliação/refinamento): {results['summary']['total_additional_tokens']}")
                
                # Calcular eficiência de refinamento com proteção contra divisão por zero
                total_tokens = results['summary']['total_tokens']
                if total_tokens > 0:
                    efficiency = (results['summary']['total_additional_tokens'] / total_tokens * 100)
                    print(f"Eficiência de refinamento: {efficiency:.2f}% tokens adicionais")
                else:
                    print("Eficiência de refinamento: N/A (sem tokens processados)")
            else:
                print("Não há estatísticas para exibir: nenhuma pergunta processada com sucesso.")
        except Exception as report_e:
            logging.error(f"Erro ao gerar relatório resumido: {report_e}")
            print("\nErro ao gerar relatório resumido.")
        
        print(f"\nTestes concluídos em {time.time() - start_time:.2f} segundos.")
        print(f"Resultados completos salvos em: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Erro durante a execução dos testes: {e}")
        print(f"\nERRO: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 