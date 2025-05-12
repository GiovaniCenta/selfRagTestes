import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import torch
import re

# Importar componentes existentes
from .retriever import retrieve_relevant_chunks
from .reranker import rank as rerank_chunks
from .llm_explainer import load_explainer_model_and_tokenizer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SELF_RAG - %(levelname)s - %(message)s')

class SelfRAG:
    def __init__(
        self,
        initial_retrieval_k: int = 10,
        reranker_n: int = 5,
        max_attempts: int = 3,
        llm_temperature: float = 0.1,
        llm_max_tokens: int = 200
    ):
        """
        Inicializa o sistema Self-RAG.
        
        Args:
            initial_retrieval_k: Número de documentos iniciais a recuperar
            reranker_n: Número de documentos após reranking
            max_attempts: Número máximo de tentativas de refinamento da resposta
            llm_temperature: Temperatura para geração LLM
            llm_max_tokens: Máximo de tokens de saída para o LLM
        """
        self.initial_retrieval_k = initial_retrieval_k
        self.reranker_n = reranker_n
        self.max_attempts = max_attempts
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        
        # Estatísticas para relatório
        self.stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "additional_tokens": 0,
            "quality_improvements": [],
            "attempts": []
        }
    
    def _format_qa_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Formata o prompt para perguntas e respostas.
        """
        context_str = "\n\n".join(f"Documento {i+1}:\n{chunk}" for i, chunk in enumerate(contexts) if chunk.strip())
        
        system_prompt_content = (
            "Você é um assistente especializado em documentos jurídicos. "
            "Responda a pergunta baseando-se apenas nos documentos fornecidos. "
            "Se a resposta não puder ser encontrada nos documentos, indique claramente que "
            "'A resposta não pode ser encontrada nos documentos fornecidos'."
        )
        
        user_content = f"Documentos:\n{context_str}\n\nPergunta: {query}\n\nResposta:"
        
        # Formatar para Llama 3.1 (similar ao usado em llm_explainer.py)
        prompt = f"<s>[INST] {system_prompt_content}\n\n{user_content.strip()} [/INST]"
        return prompt
    
    def _format_evaluation_prompt(self, query: str, answer: str) -> str:
        """
        Formata o prompt para auto-avaliação da resposta.
        """
        system_prompt_content = (
            "Você é um avaliador crítico de respostas a perguntas. "
            "Sua tarefa é analisar a resposta à pergunta fornecida e avaliar sua qualidade em uma escala de 1 a 10. "
            "Forneça primeiro a pontuação numérica (1-10) e depois uma justificativa para sua avaliação. "
            "Identifique qualquer informação ausente, incorreta ou imprecisa na resposta."
        )
        
        user_content = f"Pergunta: {query}\n\nResposta: {answer}\n\nAvaliação (primeiro forneça a pontuação de 1 a 10 e depois a justificativa):"
        
        # Formatar para Llama 3.1
        prompt = f"<s>[INST] {system_prompt_content}\n\n{user_content.strip()} [/INST]"
        return prompt
    
    def _format_refinement_prompt(self, query: str, contexts: List[str], previous_answer: str, evaluation: str) -> str:
        """
        Formata o prompt para refinamento da resposta.
        """
        context_str = "\n\n".join(f"Documento {i+1}:\n{chunk}" for i, chunk in enumerate(contexts) if chunk.strip())
        
        system_prompt_content = (
            "Você é um assistente especializado em documentos jurídicos. "
            "Sua tarefa é melhorar uma resposta anterior com base na avaliação fornecida. "
            "Use apenas as informações contidas nos documentos para gerar sua resposta melhorada."
        )
        
        user_content = (
            f"Documentos:\n{context_str}\n\n"
            f"Pergunta: {query}\n\n"
            f"Resposta anterior: {previous_answer}\n\n"
            f"Avaliação da resposta anterior: {evaluation}\n\n"
            f"Resposta melhorada:"
        )
        
        # Formatar para Llama 3.1
        prompt = f"<s>[INST] {system_prompt_content}\n\n{user_content.strip()} [/INST]"
        return prompt
    
    def _generate_llm_response(self, prompt: str) -> Tuple[str, int, int]:
        """
        Gera uma resposta do LLM para o prompt fornecido.
        
        Returns:
            Tuple contendo (resposta, tokens_entrada, tokens_saída)
        """
        model, tokenizer = load_explainer_model_and_tokenizer()
        device = model.device if hasattr(model, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False).to(device)
        input_token_count = inputs['input_ids'].shape[1]
        
        generation_kwargs = {
            "max_new_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature,
            "top_p": 0.9,
            "do_sample": True if self.llm_temperature > 0 else False,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        prompt_length = inputs.input_ids.shape[1]
        generated_text_ids = outputs[0, prompt_length:]
        llm_response = tokenizer.decode(generated_text_ids, skip_special_tokens=True).strip()
        output_token_count = len(generated_text_ids)
        
        return llm_response, input_token_count, output_token_count
    
    def _evaluate_answer(self, query: str, answer: str) -> Tuple[float, str]:
        """
        Avalia a qualidade da resposta.
        
        Returns:
            Tuple contendo (pontuação_qualidade, avaliação_textual)
        """
        evaluation_prompt = self._format_evaluation_prompt(query, answer)
        evaluation_response, eval_in_tokens, eval_out_tokens = self._generate_llm_response(evaluation_prompt)
        
        # Adicionar tokens ao total
        self.stats["total_input_tokens"] += eval_in_tokens
        self.stats["total_output_tokens"] += eval_out_tokens
        self.stats["additional_tokens"] += (eval_in_tokens + eval_out_tokens)
        
        # Extrair pontuação
        try:
            # Tentar encontrar um número entre 1 e 10 nas primeiras 20 caracteres da resposta
            score_match = re.search(r'\b([1-9]|10)\b', evaluation_response[:30])
            score = float(score_match.group(1)) if score_match else 5.0
        except Exception as e:
            logging.warning(f"Erro ao extrair pontuação da avaliação: {e}")
            score = 5.0  # Valor padrão se a extração falhar
        
        return score, evaluation_response
    
    def answer_question(self, query: str, processo_id: str = None) -> Dict[str, Any]:
        """
        Responde uma pergunta utilizando self-RAG.
        
        Args:
            query: A pergunta a ser respondida
            processo_id: Opcional - ID do processo para filtrar a busca
            
        Returns:
            Dict contendo resposta final e métricas.
        """
        start_time = time.time()
        self.stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "additional_tokens": 0,
            "quality_improvements": [],
            "attempts": []
        }
        
        # Primeira tentativa
        attempt = 1
        logging.info(f"Tentativa {attempt}: Recuperando chunks iniciais para a pergunta")
        
        try:
            # Recuperar chunks iniciais
            retrieved_context = retrieve_relevant_chunks(query, top_k=self.initial_retrieval_k)
            if not retrieved_context or not retrieved_context.get('documents') or not retrieved_context['documents'][0]:
                logging.error(f"Não foi possível recuperar informações para a pergunta: {query[:50]}...")
                return {
                    "answer": "Não foi possível recuperar informações para responder à pergunta.",
                    "final_quality_score": 0,
                    "evaluation": "Falha na recuperação de documentos relevantes.",
                    "total_attempts": 1,
                    "context_used": [],
                    "stats": self.stats,
                    "processing_time": time.time() - start_time
                }
        except Exception as e:
            logging.error(f"Erro durante a recuperação de documentos: {e}")
            return {
                "answer": f"Erro ao processar a pergunta: {str(e)}",
                "final_quality_score": 0,
                "evaluation": "Erro no sistema de recuperação.",
                "total_attempts": 0,
                "context_used": [],
                "stats": self.stats,
                "processing_time": time.time() - start_time
            }
        
        # Construir chunks para reranking
        ids = retrieved_context.get('ids', [[]])[0]
        documents = retrieved_context.get('documents', [[]])[0]
        metadatas = retrieved_context.get('metadatas', [[]])[0]
        distances = retrieved_context.get('distances', [[]])[0]
        
        initial_chunks = [
            {
                'id': ids[i], 'text': documents[i],
                'metadata': metadatas[i], 'distance': distances[i]
            } for i in range(len(ids))
        ]
        
        # Reranking
        logging.info(f"Tentativa {attempt}: Rerankeando {len(initial_chunks)} chunks")
        try:
            reranked_chunks = rerank_chunks(query, initial_chunks)
            
            if not reranked_chunks:
                return {
                    "answer": "Documentos recuperados não foram considerados relevantes após reranking.",
                    "final_quality_score": 0,
                    "evaluation": "Sem contexto relevante após reranking.",
                    "total_attempts": 1,
                    "context_used": [],
                    "stats": self.stats,
                    "processing_time": time.time() - start_time
                }
        except Exception as e:
            logging.error(f"Erro durante o reranking: {e}")
            return {
                "answer": f"Erro ao reranquear os documentos: {str(e)}",
                "final_quality_score": 0,
                "evaluation": "Erro no sistema de reranking.",
                "total_attempts": 1,
                "context_used": [],
                "stats": self.stats,
                "processing_time": time.time() - start_time
            }
        
        # Selecionar top N chunks após reranking
        selected_contexts = [chunk['text'] for chunk in reranked_chunks[:self.reranker_n]]
        
        # Gerar resposta inicial
        try:
            qa_prompt = self._format_qa_prompt(query, selected_contexts)
            current_answer, in_tokens, out_tokens = self._generate_llm_response(qa_prompt)
            
            # Atualizar estatísticas
            self.stats["total_input_tokens"] += in_tokens
            self.stats["total_output_tokens"] += out_tokens
        except Exception as e:
            logging.error(f"Erro ao gerar resposta inicial: {e}")
            return {
                "answer": f"Erro ao gerar resposta: {str(e)}",
                "final_quality_score": 0,
                "evaluation": "Erro no sistema de geração de resposta.",
                "total_attempts": 1,
                "context_used": selected_contexts,
                "stats": self.stats,
                "processing_time": time.time() - start_time
            }
        
        # Avaliar resposta inicial
        try:
            quality_score, evaluation = self._evaluate_answer(query, current_answer)
            self.stats["quality_improvements"].append(quality_score)
            self.stats["attempts"].append({
                "attempt": attempt,
                "answer": current_answer,
                "evaluation": evaluation,
                "quality_score": quality_score
            })
            
            logging.info(f"Tentativa {attempt}: Resposta inicial gerada com qualidade {quality_score}/10")
        except Exception as e:
            logging.error(f"Erro ao avaliar resposta inicial: {e}")
            # Continuar com uma pontuação padrão
            quality_score = 5.0
            evaluation = f"Erro na avaliação: {str(e)}"
            self.stats["quality_improvements"].append(quality_score)
            self.stats["attempts"].append({
                "attempt": attempt,
                "answer": current_answer,
                "evaluation": evaluation,
                "quality_score": quality_score
            })
        
        # Loop de refinamento se necessário
        while attempt < self.max_attempts and quality_score < 7.0:
            attempt += 1
            logging.info(f"Tentativa {attempt}: Refinando resposta (qualidade atual: {quality_score}/10)")
            
            try:
                # Gerar resposta refinada
                refinement_prompt = self._format_refinement_prompt(
                    query, selected_contexts, current_answer, evaluation
                )
                
                improved_answer, ref_in_tokens, ref_out_tokens = self._generate_llm_response(refinement_prompt)
                
                # Atualizar estatísticas
                self.stats["total_input_tokens"] += ref_in_tokens
                self.stats["total_output_tokens"] += ref_out_tokens
                self.stats["additional_tokens"] += (ref_in_tokens + ref_out_tokens)
                
                # Avaliar resposta refinada
                new_quality_score, new_evaluation = self._evaluate_answer(query, improved_answer)
                
                # Verificar se houve melhoria
                if new_quality_score > quality_score:
                    improvement = new_quality_score - quality_score
                    logging.info(f"Tentativa {attempt}: Melhoria na qualidade: +{improvement:.2f} pontos")
                    
                    # Atualizar para a resposta melhorada
                    current_answer = improved_answer
                    quality_score = new_quality_score
                    evaluation = new_evaluation
                else:
                    logging.info(f"Tentativa {attempt}: Sem melhoria na qualidade. Mantendo resposta anterior.")
                
                self.stats["quality_improvements"].append(new_quality_score)
                self.stats["attempts"].append({
                    "attempt": attempt,
                    "answer": improved_answer,
                    "evaluation": new_evaluation,
                    "quality_score": new_quality_score
                })
            except Exception as e:
                logging.error(f"Erro durante a tentativa {attempt} de refinamento: {e}")
                # Sair do loop se houver falha no refinamento
                break
        
        # Preparar resultado final
        processing_time = time.time() - start_time
        
        return {
            "answer": current_answer,
            "final_quality_score": quality_score,
            "evaluation": evaluation,
            "total_attempts": attempt,
            "context_used": selected_contexts,
            "stats": self.stats,
            "processing_time": processing_time
        }

# Função auxiliar para uso direto do módulo
def answer_with_self_rag(
    query: str, 
    processo_id: str = None,
    initial_k: int = 10, 
    reranker_n: int = 5, 
    max_attempts: int = 3, 
    temperature: float = 0.1,
    max_tokens: int = 200
) -> Dict[str, Any]:
    """
    Função auxiliar para responder a perguntas usando Self-RAG.
    
    Args:
        query: Pergunta a ser respondida
        processo_id: ID do processo (opcional)
        initial_k: Número de documentos iniciais
        reranker_n: Número de documentos após reranking
        max_attempts: Máximo de tentativas de melhoria
        temperature: Temperatura para geração LLM
        max_tokens: Máximo de tokens por resposta
        
    Returns:
        Dicionário com a resposta e métricas
    """
    rag = SelfRAG(
        initial_retrieval_k=initial_k,
        reranker_n=reranker_n,
        max_attempts=max_attempts,
        llm_temperature=temperature,
        llm_max_tokens=max_tokens
    )
    
    return rag.answer_question(query, processo_id)

# Função para executar testes automatizados
def run_self_rag_tests(
    output_dir: str = "results",
    initial_k: int = 10,
    reranker_n: int = 5,
    max_attempts: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 200,
    select_category: str = "",
    select_question: str = ""
) -> Dict[str, Any]:
    """
    Executa uma série de testes predefinidos para avaliar o sistema Self-RAG.
    
    Args:
        output_dir: Diretório para salvar os resultados
        initial_k: Número de documentos iniciais a recuperar
        reranker_n: Número de documentos após reranking
        max_attempts: Máximo de tentativas de refinamento
        temperature: Temperatura do LLM
        max_tokens: Máximo de tokens por resposta
        select_category: Categoria específica para testar (opcional)
        select_question: ID de pergunta específica para testar (opcional)
        
    Returns:
        Dicionário com resultados do teste
    """
    # Garantir que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Perguntas predefinidas para testes
    test_questions = {
        "acordao_733_2025": [
            {
                "id": "733_simples_1",
                "question": "Qual a principal conclusão do TCU em relação ao BNDES no Acórdão 733/2025?",
                "complexity": "simples"
            },
            {
                "id": "733_resumo_1",
                "question": "Resuma a controvérsia sobre o status do BNDES como empresa dependente da União.",
                "complexity": "média"
            },
            {
                "id": "733_contraste_1",
                "question": "Explique as contradições presentes entre as diferentes interpretações sobre os recursos utilizados pelo BNDES e seu status como dependente ou não da União.",
                "complexity": "complexa"
            },
            {
                "id": "733_especifica_1",
                "question": "O que o TCU determinou sobre a participação nos lucros e resultados (PLR) paga aos funcionários do BNDES?",
                "complexity": "média"
            },
            {
                "id": "733_analise_1",
                "question": "Por que o TCU considerou que o BNDES deveria estar sujeito ao teto remuneratório constitucional, e quais as implicações desta decisão?",
                "complexity": "complexa"
            }
        ],
        "acordao_764_2025": [
            {
                "id": "764_simples_1",
                "question": "Qual o objeto da representação no processo TC 024.887/2024-2?",
                "complexity": "simples"
            },
            {
                "id": "764_contraste_1",
                "question": "Compare as diferentes versões sobre a contratação realizada pelo Crea/SP mencionadas nos resumos.",
                "complexity": "complexa"
            },
            {
                "id": "764_especifica_1",
                "question": "Que empresa foi apontada como beneficiária de direcionamento na contratação realizada pelo Crea/SP?",
                "complexity": "simples"
            },
            {
                "id": "764_medidas_1",
                "question": "Quais foram as medidas determinadas pelo TCU em relação ao contrato do Crea/SP no Acórdão 764/2025?",
                "complexity": "média"
            },
            {
                "id": "764_analise_1",
                "question": "Analise as razões que levaram o TCU a determinar o ressarcimento pela empresa contratada pelo Crea/SP.",
                "complexity": "complexa"
            }
        ],
        "questoes_complexas": [
            {
                "id": "complexa_comparativa_1",
                "question": "Compare os processos de tomada de decisão do TCU nos Acórdãos 733/2025 e 764/2025, identificando semelhanças e diferenças na aplicação das normas de controle.",
                "complexity": "muito complexa"
            },
            {
                "id": "complexa_733_1",
                "question": "Avalie os argumentos contraditórios sobre o BNDES ser dependente ou não da União, considerando as fontes de recursos mencionadas nos resumos.",
                "complexity": "muito complexa"
            },
            {
                "id": "complexa_764_1",
                "question": "Analise como as decisões contraditórias apresentadas nos resumos sobre o contrato do Crea/SP poderiam ser conciliadas para formar uma interpretação coerente do caso.",
                "complexity": "muito complexa"
            }
        ]
    }
    
    # Aplicar filtros por categoria se especificado
    if select_category:
        if select_category in test_questions:
            filtered_questions = {select_category: test_questions[select_category]}
            test_questions = filtered_questions
            print(f"Filtrando para categoria: {select_category}")
        else:
            logging.warning(f"Categoria '{select_category}' não encontrada. Usando todas as categorias.")
    
    # Aplicar filtro por ID de pergunta específica
    if select_question:
        found_question = False
        for category in list(test_questions.keys()):
            filtered_category = []
            for q in test_questions[category]:
                if q["id"] == select_question:
                    filtered_category.append(q)
                    found_question = True
                    print(f"Filtrando para pergunta específica: {select_question}")
            
            if filtered_category:
                test_questions[category] = filtered_category
            else:
                # Remover categoria se não contiver a pergunta específica
                test_questions.pop(category)
        
        if not found_question:
            logging.warning(f"Pergunta com ID '{select_question}' não encontrada. Usando todas as perguntas.")
    
    # Inicializar resultados
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "config": {
                "initial_k": initial_k,
                "reranker_n": reranker_n,
                "max_attempts": max_attempts,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "select_category": select_category,
                "select_question": select_question
            }
        },
        "results_by_category": {},
        "summary": {
            "total_questions": 0,
            "total_tokens": 0,
            "total_additional_tokens": 0,
            "avg_quality_score": 0,
            "avg_attempts": 0,
            "processing_time": 0,
            "successful_questions": 0,
            "failed_questions": 0
        }
    }
    
    # Processar cada categoria de perguntas
    total_start_time = time.time()
    total_questions = 0
    successful_questions = 0
    failed_questions = 0
    all_quality_scores = []
    all_attempts = []
    total_tokens = 0
    total_additional_tokens = 0
    
    for category, questions in test_questions.items():
        results["results_by_category"][category] = []
        
        print(f"\n{'='*20} Testando categoria: {category} {'='*20}\n")
        
        for q_data in questions:
            q_id = q_data["id"]
            query = q_data["question"]
            complexity = q_data["complexity"]
            
            print(f"Processando pergunta {q_id}: {query}")
            print(f"Complexidade: {complexity}")
            
            try:
                # Processar a pergunta
                qa_result = answer_with_self_rag(
                    query=query,
                    initial_k=initial_k,
                    reranker_n=reranker_n,
                    max_attempts=max_attempts,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Adicionar metadados da pergunta
                qa_result["question_id"] = q_id
                qa_result["question"] = query
                qa_result["complexity"] = complexity
                
                # Armazenar resultado
                results["results_by_category"][category].append(qa_result)
                
                # Acumular estatísticas
                total_questions += 1
                
                # Verificar se a pergunta foi processada com sucesso
                if "final_quality_score" in qa_result and qa_result["final_quality_score"] > 0:
                    successful_questions += 1
                    all_quality_scores.append(qa_result["final_quality_score"])
                    all_attempts.append(qa_result["total_attempts"])
                    
                    tokens_in_this_query = qa_result["stats"]["total_input_tokens"] + qa_result["stats"]["total_output_tokens"]
                    total_tokens += tokens_in_this_query
                    total_additional_tokens += qa_result["stats"]["additional_tokens"]
                    
                    # Mostrar resultado da pergunta
                    print(f"\nResposta final: {qa_result['answer'][:150]}...")
                    print(f"Qualidade: {qa_result['final_quality_score']}/10")
                    print(f"Tentativas: {qa_result['total_attempts']}/{max_attempts}")
                    print(f"Tokens: {tokens_in_this_query} (Adicionais: {qa_result['stats']['additional_tokens']})")
                else:
                    # Se houve erro, a pergunta não foi processada completamente
                    failed_questions += 1
                    print(f"\nProcessamento incompleto ou com falha: {qa_result.get('answer', 'Erro desconhecido')[:150]}...")
                
                print(f"{'='*60}\n")
                
            except Exception as e:
                logging.error(f"Erro processando pergunta {q_id}: {e}")
                print(f"ERRO em {q_id}: {e}")
                # Registrar erro no resultado
                failed_questions += 1
                total_questions += 1
                error_result = {
                    "question_id": q_id,
                    "question": query,
                    "complexity": complexity,
                    "error": str(e),
                    "answer": f"Erro: {str(e)}",
                    "final_quality_score": 0,
                    "total_attempts": 0
                }
                results["results_by_category"][category].append(error_result)
    
    # Finalizar estatísticas
    total_processing_time = time.time() - total_start_time
    
    # Calcular médias (evitando divisão por zero)
    avg_quality = sum(all_quality_scores) / max(len(all_quality_scores), 1) if all_quality_scores else 0
    avg_attempts = sum(all_attempts) / max(len(all_attempts), 1) if all_attempts else 0
    
    # Atualizar resumo
    results["summary"]["total_questions"] = total_questions
    results["summary"]["successful_questions"] = successful_questions
    results["summary"]["failed_questions"] = failed_questions
    results["summary"]["total_tokens"] = total_tokens
    results["summary"]["total_additional_tokens"] = total_additional_tokens
    results["summary"]["avg_quality_score"] = avg_quality
    results["summary"]["avg_attempts"] = avg_attempts
    results["summary"]["processing_time"] = total_processing_time
    
    # Salvar resultados completos em JSON
    output_file = os.path.join(output_dir, f"self_rag_test_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResultados completos salvos em: {output_file}")
    except Exception as e:
        logging.error(f"Erro ao salvar resultados: {e}")
        print(f"Erro ao salvar resultados: {e}")
    
    # Gerar relatório resumido
    print(f"\n{'='*30} RELATÓRIO RESUMIDO {'='*30}")
    print(f"Total de perguntas: {total_questions}")
    print(f"Perguntas processadas com sucesso: {successful_questions}")
    print(f"Perguntas com falha: {failed_questions}")
    print(f"Tempo total de processamento: {total_processing_time:.2f}s")
    
    if successful_questions > 0:
        print(f"Qualidade média das respostas: {avg_quality:.2f}/10")
        print(f"Média de tentativas por pergunta: {avg_attempts:.2f}/{max_attempts}")
        print(f"Total de tokens consumidos: {total_tokens}")
        print(f"Tokens adicionais (avaliação/refinamento): {total_additional_tokens}")
        
        # Calcular eficiência de refinamento (evitando divisão por zero)
        if total_tokens > 0:
            efficiency = (total_additional_tokens / total_tokens) * 100
            print(f"Eficiência de refinamento: {efficiency:.2f}% tokens adicionais")
        else:
            print("Eficiência de refinamento: N/A (sem tokens processados)")
    else:
        print("Não há estatísticas de qualidade para exibir: nenhuma pergunta processada com sucesso.")
    
    # Análise por complexidade
    complexity_data = {}
    for category in results["results_by_category"]:
        for item in results["results_by_category"][category]:
            if "complexity" in item and "final_quality_score" in item:
                complexity = item["complexity"]
                if complexity not in complexity_data:
                    complexity_data[complexity] = {
                        "count": 0,
                        "quality_scores": [],
                        "attempts": []
                    }
                complexity_data[complexity]["count"] += 1
                
                # Só adicionar à estatística se tiver uma pontuação válida
                if item["final_quality_score"] > 0:
                    complexity_data[complexity]["quality_scores"].append(item["final_quality_score"])
                    complexity_data[complexity]["attempts"].append(item.get("total_attempts", 0))
    
    if complexity_data:
        print("\nAnálise por complexidade:")
        for complexity, data in complexity_data.items():
            # Calcular médias (evitando divisão por zero)
            quality_scores = data["quality_scores"]
            attempts = data["attempts"]
            
            avg_q = sum(quality_scores) / max(len(quality_scores), 1) if quality_scores else 0
            avg_a = sum(attempts) / max(len(attempts), 1) if attempts else 0
            
            success_count = len(quality_scores)
            total_count = data["count"]
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            print(f"- {complexity.upper()}: {total_count} perguntas ({success_count} sucesso, {success_rate:.1f}%), " +
                  f"Qualidade média: {avg_q:.2f}/10, Tentativas médias: {avg_a:.2f}")
    
    return results


if __name__ == "__main__":
    try:
        # Se executado diretamente, rodar os testes automatizados
        print("\n" + "="*20 + " Iniciando Testes Automatizados do Self-RAG " + "="*20)
        run_self_rag_tests(
            initial_k=10,
            reranker_n=5,
            max_attempts=3,
            temperature=0.1,
            max_tokens=250
        )
    except Exception as e:
        print(f"Erro nos testes automatizados: {e}") 