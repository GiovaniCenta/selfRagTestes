import streamlit as st
import os
import sys
import logging
import time
import tempfile
import json
from pathlib import Path

# Configure logging for Streamlit
log_level = logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - APP - %(levelname)s - %(message)s')

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

# Import project modules
# Wrap imports in try/except for clearer error messages
try:
    from src.data_loader import load_and_prepare_data, load_document, split_text_into_paragraphs
    from src.indexer import get_embedding_model, create_or_update_index, CHROMA_PERSIST_DIR
    from src.retriever import retrieve_relevant_chunks
    from src.reranker import rerank_chunks, RERANKER_SCORE_THRESHOLD, RERANKER_SELECT_TOP_N
    from src.validator import validate_claim_with_llm
    from src.self_rag import answer_with_self_rag, SelfRAG
    INITIAL_RETRIEVAL_TOP_K = 5
except ImportError as e:
    st.error(f"Erro ao importar módulos necessários: {e}. Verifique se 'src' está no PYTHONPATH.")
    st.stop() # Stop the app if core modules are missing

# === Page Functions ===

def page_create_index():
    st.header("⚙️ Criar/Atualizar Índice Vetorial")
    st.write(f"O índice será salvo em: `{CHROMA_PERSIST_DIR}`")

    uploaded_files = st.file_uploader(
        "Selecione um ou mais arquivos de Acórdãos (.pdf ou .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True # Allows multiple files upload
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} arquivo(s) selecionado(s):")
        for f in uploaded_files:
            st.write(f" - `{f.name}`")

        if st.button("Iniciar Indexação", key="start_indexing_multi"):
            st.markdown("--- R E S U L T A D O ---")
            progress_bar = st.progress(0, text="Iniciando...")
            status_placeholder = st.empty()

            all_acordao_chunks = []
            processed_files = 0
            failed_files = []
            temp_files_to_remove = []

            try:
                total_files = len(uploaded_files)
                # Process each uploaded file
                for i, uploaded_file in enumerate(uploaded_files):
                    file_index = i + 1
                    file_progress_start = int((i / total_files) * 50) # First 50% for loading
                    file_progress_end = int(((i + 1) / total_files) * 50)
                    progress_bar.progress(file_progress_start, text=f"Processando arquivo {file_index}/{total_files}: {uploaded_file.name}")
                    status_placeholder.info(f"1/3 - Carregando e preparando {uploaded_file.name}...")

                    # Save uploaded file temporarily
                    tmp_file_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                            temp_files_to_remove.append(tmp_file_path) # Keep track for cleanup
                            logging.info(f"Arquivo temporário salvo em: {tmp_file_path}")

                        # Load data for current file
                        acordao_chunks, _ = load_and_prepare_data(tmp_file_path, "") # Resumo not needed here

                        if acordao_chunks is None:
                            logging.warning(f"Falha ao carregar ou processar o documento: {uploaded_file.name}")
                            failed_files.append(uploaded_file.name)
                        else:
                            all_acordao_chunks.extend(acordao_chunks)
                            processed_files += 1
                            status_placeholder.info(f"1/3 - {uploaded_file.name} carregado: {len(acordao_chunks)} chunks encontrados.")

                    except Exception as file_e:
                        logging.error(f"Erro processando arquivo {uploaded_file.name}: {file_e}", exc_info=True)
                        failed_files.append(f"{uploaded_file.name} (Erro: {file_e})")
                    finally:
                        progress_bar.progress(file_progress_end, text=f"Arquivo {file_index}/{total_files} processado.")

                status_placeholder.info(f"1/3 - Leitura de arquivos concluída. {processed_files}/{total_files} processados com sucesso. Chunks totais: {len(all_acordao_chunks)}.")
                progress_bar.progress(50, text="Leitura de arquivos concluída.")

                if not all_acordao_chunks:
                    raise ValueError("Nenhum chunk extraído de nenhum dos arquivos fornecidos.")

                # Load Embedding Model (once for all chunks)
                status_placeholder.info("2/3 - Carregando modelo de embedding (pode demorar)...")
                progress_bar.progress(60, text="Carregando modelo embedding...")
                embedding_model = get_embedding_model()
                if embedding_model is None:
                    raise ValueError("Falha ao carregar o modelo de embedding.")
                status_placeholder.info("2/3 - Modelo de embedding carregado.")
                progress_bar.progress(80, text="Modelo embedding carregado.")

                # Create/Update Index (with all collected chunks)
                status_placeholder.info(f"3/3 - Indexando {len(all_acordao_chunks)} chunks totais (pode demorar)...")
                progress_bar.progress(85, text="Indexando chunks...")
                create_or_update_index(all_acordao_chunks, embedding_model)
                status_placeholder.info("3/3 - Indexação concluída.")
                progress_bar.progress(95, text="Indexação concluída.")

                # Finish
                progress_bar.progress(100, text="Concluído!")
                success_message = f"🎉 Índice criado/atualizado com sucesso em `{CHROMA_PERSIST_DIR}` com chunks de {processed_files} arquivo(s)."
                if failed_files:
                    success_message += f"\n⚠️ Arquivos que falharam: {', '.join(failed_files)}"
                status_placeholder.success(success_message)

            except Exception as e:
                logging.error(f"Erro geral durante a indexação: {e}", exc_info=True)
                error_message = f"Erro durante a indexação: {e}"
                if failed_files:
                    error_message += f"\nArquivos que falharam antes do erro principal: {', '.join(failed_files)}"
                status_placeholder.error(error_message)
                progress_bar.progress(100, text="Falha!")
            finally:
                # Clean up the temporary files
                for f_path in temp_files_to_remove:
                    if os.path.exists(f_path):
                        try:
                            os.remove(f_path)
                            logging.info(f"Arquivo temporário removido: {f_path}")
                        except Exception as del_e:
                            logging.warning(f"Falha ao remover arquivo temporário {f_path}: {del_e}")

def page_retrieve_chunks():
    st.header("🔍 Consultar/Recuperar Chunks")
    st.write(f"Consulta será feita no índice em: `{CHROMA_PERSIST_DIR}`")

    query = st.text_area("Digite sua consulta/alegação aqui:", height=100)

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.number_input("Docs. Iniciais (k):", min_value=1, max_value=20, value=INITIAL_RETRIEVAL_TOP_K)
    with col2:
        use_reranker = st.checkbox("Usar Re-ranker?", value=True)
    with col3:
        top_n_rerank = st.number_input("Top N (Pós-Rerank):", min_value=1, max_value=top_k, value=RERANKER_SELECT_TOP_N, disabled=not use_reranker)

    if st.button("Recuperar Chunks", key="retrieve_chunks_btn"):
        if not query:
            st.warning("Por favor, digite uma consulta.")
            st.stop()
        if not os.path.exists(CHROMA_PERSIST_DIR):
             st.error(f"Diretório do índice ChromaDB não encontrado em {CHROMA_PERSIST_DIR}. Crie o índice primeiro.")
             st.stop()

        st.markdown("--- R E S U L T A D O ---")
        final_chunks = []
        with st.spinner(f"Recuperando {top_k} chunks iniciais..."):
            try:
                retrieved_context = retrieve_relevant_chunks(query, top_k=top_k)
                if retrieved_context is None or not retrieved_context.get('documents') or not retrieved_context['documents'][0]:
                    st.warning("Recuperação inicial falhou ou não retornou documentos.")
                    initial_chunks_data = []
                else:
                    # Reconstruct list of dicts for processing - could be simpler?
                    ids = retrieved_context.get('ids', [[]])[0]
                    documents = retrieved_context.get('documents', [[]])[0]
                    metadatas = retrieved_context.get('metadatas', [[]])[0]
                    distances = retrieved_context.get('distances', [[]])[0]
                    initial_chunks_data = [
                        {
                            'id': ids[i], 'text': documents[i],
                            'metadata': metadatas[i], 'distance': distances[i]
                        } for i in range(len(ids))
                    ]
                    st.info(f"{len(initial_chunks_data)} chunks recuperados inicialmente.")
            except Exception as e:
                st.error(f"Erro durante a recuperação inicial: {e}")
                initial_chunks_data = []
                st.stop()

        if use_reranker and initial_chunks_data:
            with st.spinner(f"Re-rankeando {len(initial_chunks_data)} chunks..."):
                try:
                    reranked_chunks = rerank_chunks(query, initial_chunks_data)
                    if not reranked_chunks:
                         st.warning("Re-ranking não retornou resultados.")
                         final_chunks = []
                    else:
                         final_chunks = reranked_chunks[:top_n_rerank] # Select top N
                         st.info(f"{len(final_chunks)} chunks selecionados após re-ranking (Top {top_n_rerank}).")
                except Exception as e:
                     st.error(f"Erro durante o re-ranking: {e}")
                     final_chunks = [] # Fallback to empty on error
        elif initial_chunks_data:
            # Using initial retrieval results without the reranker
            final_chunks = initial_chunks_data
            st.info("Exibindo resultados da recuperação inicial (sem re-ranking).")
        else:
            final_chunks = []
            # Message already shown if initial retrieval failed

        # Display final chunks
        if final_chunks:
            st.subheader(f"Top {len(final_chunks)} Chunks Relevantes:")
            for i, chunk in enumerate(final_chunks):
                # Show first few expanders open
                is_expanded = i < 3
                with st.expander(f"Chunk {i+1} (Score Rerank: {chunk.get('rerank_score', 'N/A'):.4f} / Dist Inicial: {chunk.get('distance', 'N/A'):.4f})", expanded=is_expanded):
                    st.markdown(f"**Texto:**\n{chunk.get('text', '')}")
                    st.json(chunk.get('metadata', {}))
        else:
            st.warning("Nenhum chunk para exibir.")


def page_validate_claims():
    st.header("✔️ Validar Alegações do Resumo")
    st.write(f"Consulta será feita no índice em: `{CHROMA_PERSIST_DIR}`")

    # Input Options: File or Text Area
    st.subheader("Fonte das Alegações")
    input_method = st.radio("Selecione o método de entrada:", ("Upload Arquivo .txt", "Digitar Alegação Única"), horizontal=True, key="claim_input_method")

    # Initialize input variables
    resumo_file = None
    single_claim_text = ""

    if input_method == "Upload Arquivo .txt":
        resumo_file = st.file_uploader("Selecione o arquivo do Resumo (.txt)", type=["txt"], key="resumo_uploader")
        if resumo_file:
             st.info(f"Arquivo de resumo selecionado: `{resumo_file.name}`")
    elif input_method == "Digitar Alegação Única":
        single_claim_text = st.text_area("Digite a alegação única aqui:", height=150, key="single_claim_area")
        if single_claim_text:
            st.info("Alegação única digitada.")

    st.subheader("Parâmetros da Validação (Opcional)")
    col1, col2 = st.columns(2)
    with col1:
        ret_top_k = st.number_input("Recuperação Inicial (k)", min_value=1, max_value=20, value=INITIAL_RETRIEVAL_TOP_K, key="val_ret_k")
        rerank_top_n = st.number_input("Seleção Pós-Rerank (N)", min_value=1, max_value=ret_top_k, value=RERANKER_SELECT_TOP_N, key="val_rerank_n")
        llm_max_tokens = st.number_input("Max Tokens LLM", min_value=20, max_value=512, value=100, step=10, key="val_max_tokens")
    with col2:
        use_reranker_val = st.checkbox("Usar Re-ranker?", value=True, key="val_use_reranker")
        rerank_threshold = st.slider("Threshold Score Reranker", min_value=0.0, max_value=1.0, value=RERANKER_SCORE_THRESHOLD, step=0.05, format="%.2f", key="val_rerank_thresh", disabled=not use_reranker_val)
        llm_temp = st.slider("Temperatura LLM", min_value=0.0, max_value=1.0, value=0.1, step=0.05, format="%.2f", key="val_temp")

    # Check if any input is ready before showing the button
    if resumo_file is not None or single_claim_text:
        if st.button("Iniciar Validação", key="start_validation"):
            if not os.path.exists(CHROMA_PERSIST_DIR):
                 st.error(f"Diretório do índice ChromaDB não encontrado em {CHROMA_PERSIST_DIR}. Crie o índice primeiro.")
                 st.stop()

            st.markdown("--- R E S U L T A D O   D A   V A L I D A Ç Ã O ---")
            results_placeholder = st.empty()
            metrics_placeholder = st.empty()
            results_list = []

            # Load claims based on input method
            claims = []
            error_loading_claims = None

            if input_method == "Digitar Alegação Única":
                if single_claim_text:
                    claims = [single_claim_text.strip()] # List with the single claim
                    logging.info("Processing single claim from text area.")
                else:
                    error_loading_claims = "Por favor, digite uma alegação na caixa de texto."
            elif input_method == "Upload Arquivo .txt":
                if resumo_file is not None:
                    try:
                        resumo_text = resumo_file.getvalue().decode("utf-8")
                        claims = split_text_into_paragraphs(resumo_text) # Uses robust splitting now
                        logging.info(f"Processing {len(claims)} claims from uploaded file: {resumo_file.name}")
                        if not claims:
                            error_loading_claims = "Nenhuma alegação encontrada no arquivo de resumo. Verifique o formato (parágrafos separados por linhas em branco)."
                    except Exception as e:
                        logging.error(f"Erro ao ler ou processar o arquivo de resumo: {e}", exc_info=True)
                        error_loading_claims = f"Erro ao ler ou processar o arquivo de resumo: {e}"
                else:
                    error_loading_claims = "Por favor, faça o upload de um arquivo de resumo .txt."
            else:
                # This case should not happen with the radio button
                error_loading_claims = "Método de entrada inválido selecionado."

            # Stop if claims could not be loaded
            if error_loading_claims:
                st.error(error_loading_claims)
                st.stop()
            if not claims:
                 st.warning("Nenhuma alegação para processar.")
                 st.stop()

            st.info(f"Iniciando validação para {len(claims)} alegação(ns)...")

            metrics_agg = {
                "total_claims": len(claims), # Set total claims here
                "processed_claims": 0,
                "retrieval_failures": 0,
                "llm_validation_failures": 0,
                "llm_parsing_failures": 0,
                "total_latency_seconds": 0,
                "total_retrieval_latency": 0,
                "total_rerank_latency": 0,
                "total_llm_latency": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "claims_with_no_reranked_context": 0
            }

            overall_start_time = time.time()
            progress_bar = st.progress(0)
            # Process each claim
            for i, claim_text in enumerate(claims):
                claim_idx = i + 1
                status_text = f"Processando alegação {claim_idx}/{len(claims)}: {claim_text[:50]}..."
                progress_bar.progress(int((i / len(claims)) * 100), text=status_text)
                results_placeholder.info(status_text)

                claim_start_time = time.time()
                retrieval_time, rerank_time, llm_time = 0, 0, 0
                status, justification = "Erro", "Processo não iniciado"
                input_tokens, output_tokens = 0, 0

                try:
                    # 1. Retrieve
                    retrieval_start = time.time()
                    retrieved_context = retrieve_relevant_chunks(claim_text, top_k=ret_top_k)
                    retrieval_time = time.time() - retrieval_start
                    metrics_agg["total_retrieval_latency"] += retrieval_time

                    if retrieved_context is None or not retrieved_context.get('documents') or not retrieved_context['documents'][0]:
                        status, justification = "Erro", "Falha na recuperação inicial."
                        metrics_agg["retrieval_failures"] += 1
                        metrics_agg["llm_validation_failures"] += 1
                        selected_chunks_for_llm = [] # No context for LLM if retrieval fail
                    else:
                        # Reconstruct chunks list for reranker
                        ids = retrieved_context.get('ids', [[]])[0]
                        docs = retrieved_context.get('documents', [[]])[0]
                        metas = retrieved_context.get('metadatas', [[]])[0]
                        dists = retrieved_context.get('distances', [[]])[0]
                        initial_chunks = [
                            {'id': ids[j], 'text': docs[j], 'metadata': metas[j], 'distance': dists[j]}
                            for j in range(len(ids))
                        ]

                        # 2. Rerank (if enabled)
                        if use_reranker_val:
                            rerank_start = time.time()
                            reranked_chunks = rerank_chunks(claim_text, initial_chunks)
                            rerank_time = time.time() - rerank_start
                            metrics_agg["total_rerank_latency"] += rerank_time
                            if not reranked_chunks:
                                status, justification = "Erro", "Re-ranking falhou."
                                metrics_agg["llm_validation_failures"] += 1
                                selected_chunks_for_llm = []
                            else:
                                # Select top N and apply threshold
                                selected_chunks_for_llm = []
                                for chunk in reranked_chunks[:rerank_top_n]:
                                    if chunk['rerank_score'] >= rerank_threshold:
                                        selected_chunks_for_llm.append(chunk['text'])
                        else:
                            # If not reranking, use top N from initial retrieval
                            selected_chunks_for_llm = [chunk['text'] for chunk in initial_chunks[:rerank_top_n]]

                        # 3. Validate (only if we have context)
                        if not selected_chunks_for_llm:
                            if use_reranker_val:
                                 status, justification = "Erro", f"Nenhum chunk passou no threshold do reranker (>={rerank_threshold})."
                                 metrics_agg["claims_with_no_reranked_context"] += 1
                            else:
                                 # Should not happen if initial retrieval worked, but maybe good for safety
                                 status, justification = "Erro", "Nenhum contexto selecionado para LLM."
                            metrics_agg["llm_validation_failures"] += 1
                        else:
                             # Construct context for the validator
                             context_for_validator = {'documents': [selected_chunks_for_llm]}
                             llm_start = time.time()
                             result = validate_claim_with_llm(claim_text, context_for_validator, max_new_tokens=llm_max_tokens, temperature=llm_temp)
                             llm_time = time.time() - llm_start
                             metrics_agg["total_llm_latency"] += llm_time
                             if result is None:
                                 status, justification = "Erro", "Falha na validação LLM (retornou None)."
                                 metrics_agg["llm_validation_failures"] += 1
                             else:
                                 status = result.get("Resultado", "Erro")
                                 justification = result.get("Justificativa", "Erro parsing.")
                                 input_tokens = result.get("input_tokens", 0)
                                 output_tokens = result.get("output_tokens", 0)
                                 metrics_agg["total_input_tokens"] += input_tokens
                                 metrics_agg["total_output_tokens"] += output_tokens
                                 if status == "Erro":
                                      metrics_agg["llm_validation_failures"] += 1
                                      # Check if it looks like a parsing error
                                      if "parsear" in justification or "Formato inesperado" in justification:
                                           metrics_agg["llm_parsing_failures"] += 1

                except Exception as e:
                    logging.error(f"Erro inesperado processando alegação {claim_idx}: {e}", exc_info=True)
                    status, justification = "Erro", f"Erro inesperado: {e}"
                    metrics_agg["llm_validation_failures"] += 1 # Count unexpected errors as LLM failures also
                finally:
                    metrics_agg["processed_claims"] += 1
                    results_list.append({
                        "#": claim_idx,
                        "Alegação": claim_text,
                        "Status": status,
                        "Justificativa/Contexto": justification
                    })
                    # Update table dynamically as results come in
                    results_placeholder.dataframe(results_list, use_container_width=True)

            # Finalize progress and metrics display
            progress_bar.progress(100, text="Validação Concluída!")
            metrics_agg["total_latency_seconds"] = time.time() - overall_start_time
            st.success("Validação concluída!")
            metrics_placeholder.subheader("Métricas da Execução")

            processed_claims = metrics_agg['processed_claims']
            if processed_claims > 0:
                 metrics_placeholder.markdown(f"""**Latência:**
                 - Total: {metrics_agg['total_latency_seconds']:.2f}s
                 - Média/Alegação: {metrics_agg['total_latency_seconds']/processed_claims:.2f}s
                 (Recuperação: {metrics_agg['total_retrieval_latency']/processed_claims:.2f}s, Rerank: {metrics_agg['total_rerank_latency']/processed_claims:.2f}s, LLM: {metrics_agg['total_llm_latency']/processed_claims:.2f}s)""")

                 metrics_placeholder.markdown(f"""**Tokens LLM:**
                 - Total: {metrics_agg['total_input_tokens'] + metrics_agg['total_output_tokens']}
                 (Entrada: {metrics_agg['total_input_tokens']}, Saída: {metrics_agg['total_output_tokens']})
                 - Média/Alegação: (Entrada: {metrics_agg['total_input_tokens']/processed_claims:.1f}, Saída: {metrics_agg['total_output_tokens']/processed_claims:.1f})""")

                 llm_failures = metrics_agg['llm_validation_failures']
                 success_rate = ((processed_claims - llm_failures) / processed_claims * 100) if processed_claims else 0
                 metrics_placeholder.markdown(f"""**Confiabilidade:**
                 - Taxa Sucesso: {success_rate:.1f}%
                 - Falhas Recuperação: {metrics_agg['retrieval_failures']}
                 - Sem Ctx Pós-Rerank: {metrics_agg['claims_with_no_reranked_context']}
                 - Falhas LLM Total: {llm_failures} (Parsing: {metrics_agg['llm_parsing_failures']})""")
            else:
                 metrics_placeholder.warning("Nenhuma alegação processada.")

def page_self_rag_qa():
    """Página para o sistema Self-RAG de perguntas e respostas."""
    st.header("🤖 Perguntas e Respostas com Self-RAG")
    st.write(f"Sistema inteligente de perguntas e respostas com auto-refinamento.")
    st.write(f"Consulta será feita no índice em: `{CHROMA_PERSIST_DIR}`")
    
    if not os.path.exists(CHROMA_PERSIST_DIR):
        st.error(f"Diretório do índice ChromaDB não encontrado em {CHROMA_PERSIST_DIR}. Crie o índice primeiro.")
        st.stop()
    
    # Interface para pergunta
    st.subheader("Faça sua pergunta")
    query = st.text_area("Digite sua pergunta sobre o documento:", height=100)
    
    # Configurações avançadas
    with st.expander("Configurações avançadas", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            initial_k = st.number_input("Documentos iniciais (k):", min_value=1, max_value=20, value=10)
            reranker_n = st.number_input("Documentos após reranking (n):", min_value=1, max_value=initial_k, value=5)
            process_id_filter = st.text_input("Filtrar por ID do processo (opcional):", value="")
        with col2:
            max_attempts = st.number_input("Máximo de tentativas de refinamento:", min_value=1, max_value=5, value=3)
            temperature = st.slider("Temperatura do LLM:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            max_tokens = st.number_input("Máximo de tokens por resposta:", min_value=50, max_value=500, value=200, step=50)
    
    # Botão para processar
    if st.button("Processar pergunta", key="process_self_rag_btn"):
        if not query:
            st.warning("Por favor, digite uma pergunta.")
            st.stop()
        
        # Mostrar progresso
        progress_container = st.container()
        result_container = st.container()
        
        with progress_container:
            st.info("Processando sua pergunta com Self-RAG...")
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Callback para atualizar o progresso
            def progress_callback(stage, attempt, message, progress_value):
                progress_bar.progress(progress_value)
                progress_text.info(f"Estágio: {stage} | Tentativa: {attempt} | {message}")
            
            # Processar pergunta
            start_time = time.time()
            try:
                # Preparar os argumentos
                qa_args = {
                    "query": query,
                    "processo_id": process_id_filter if process_id_filter else None,
                    "initial_k": initial_k,
                    "reranker_n": reranker_n,
                    "max_attempts": max_attempts,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Lógica para atualizar o progresso baseado na fase
                progress_callback("Recuperação", 1, "Recuperando documentos relevantes...", 10)
                
                # Executar Self-RAG
                result = answer_with_self_rag(**qa_args)
                
                # Atualizar progresso para conclusão
                progress_callback("Concluído", result["total_attempts"], "Processamento finalizado!", 100)
                
            except Exception as e:
                st.error(f"Erro ao processar pergunta: {e}")
                return
            
            processing_time = time.time() - start_time
        
        # Mostrar resultados
        with result_container:
            st.markdown("---")
            st.subheader("Resposta")
            
            # Exibir a resposta principal
            st.markdown(f"**Pergunta:** {query}")
            st.markdown(f"**Resposta final:**")
            st.markdown(f"{result['answer']}")
            
            # Métricas e estatísticas
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Qualidade", f"{result['final_quality_score']:.1f}/10")
            with col2:
                st.metric("Tentativas", f"{result['total_attempts']}/{max_attempts}")
            with col3:
                tokens_total = result['stats']['total_input_tokens'] + result['stats']['total_output_tokens']
                st.metric("Tokens Totais", tokens_total)
            with col4:
                st.metric("Tempo (s)", f"{processing_time:.2f}")
            
            # Detalhes sobre o processo
            with st.expander("Detalhes do processo de refinamento", expanded=False):
                st.markdown("### Histórico de tentativas")
                
                for i, attempt_data in enumerate(result['stats']['attempts']):
                    attempt_num = attempt_data['attempt']
                    quality = attempt_data['quality_score']
                    
                    st.markdown(f"**Tentativa {attempt_num}** (Qualidade: {quality:.1f}/10)")
                    st.markdown(f"**Resposta:**\n{attempt_data['answer']}")
                    st.markdown(f"**Avaliação:**\n{attempt_data['evaluation']}")
                    st.markdown("---")
                
                st.markdown("### Métricas detalhadas")
                st.json(
                    {
                        "tokens_entrada": result['stats']['total_input_tokens'],
                        "tokens_saida": result['stats']['total_output_tokens'],
                        "tokens_adicionais": result['stats']['additional_tokens'],
                        "progresso_qualidade": result['stats']['quality_improvements'],
                        "tempo_processamento_segundos": processing_time
                    }
                )
            
            # Contexto utilizado
            with st.expander("Contexto utilizado na resposta", expanded=False):
                for i, context in enumerate(result['context_used']):
                    st.markdown(f"**Documento {i+1}:**")
                    st.markdown(f"{context}")
                    st.markdown("---")
            
            # Opção para salvar resultado
            if st.button("Salvar resultado em JSON"):
                try:
                    # Criar nome do arquivo com timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"self_rag_result_{timestamp}.json"
                    
                    # Garantir que o diretório existe
                    os.makedirs("results", exist_ok=True)
                    
                    # Salvar resultado
                    with open(os.path.join("results", filename), "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"Resultado salvo em results/{filename}")
                except Exception as e:
                    st.error(f"Erro ao salvar resultado: {e}")

# === Main App Structure ===
st.set_page_config(page_title="Validador de Acórdãos", layout="wide")
st.title("⚖️ Validador de Resumos de Acórdãos TCU")

# --- Sidebar Navigation ---
st.sidebar.title("Navegação")
page_options = ["Criar Índice", "Consultar Chunks", "Validar Alegações", "Perguntas e Respostas"]
selected_page = st.sidebar.radio("Selecione a Ação:", page_options)
st.sidebar.markdown("--- ")
st.sidebar.info("Aplicação para validar alegações de resumos contra documentos originais de acórdãos.")

# --- Page Routing ---
if selected_page == "Criar Índice":
    page_create_index()
elif selected_page == "Consultar Chunks":
    page_retrieve_chunks()
elif selected_page == "Validar Alegações":
    page_validate_claims()
elif selected_page == "Perguntas e Respostas":
    page_self_rag_qa()
else:
    st.error("Página não encontrada.") 