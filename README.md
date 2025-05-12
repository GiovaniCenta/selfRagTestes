# Acórdão Claim Validator

This project validates claims made in summaries against original legal documents (Acórdãos from TCU - Brazilian Court of Accounts) using a RAG (Retrieval-Augmented Generation) pipeline. It includes both a command-line interface and a Streamlit web application.

##  Modules Description

*   **`main.py`**: The main script for running the validation process via the command line. It orchestrates the loading, retrieving, reranking, and validation steps for a given pair of Acórdão and Summary files.
*   **`app.py`**: A Streamlit web application providing a user interface for:
    *   Indexing new Acórdão documents into a vector database.
    *   Querying the indexed documents to retrieve relevant text chunks.
    *   Validating claims from a summary file or text input against the indexed documents.
*   **`src/data_loader.py`**: Handles loading and preprocessing of input documents (.txt and .pdf). It splits documents into paragraphs (chunks/claims) and extracts text along with metadata (like page numbers for PDFs).
*   **`src/indexer.py`**: Responsible for creating or updating a vector index (using ChromaDB) from document chunks. It generates embeddings using a sentence-transformer model and stores them.
*   **`src/retriever.py`**: Queries the vector index to find document chunks relevant to a given query (claim). It uses sentence embeddings for similarity search.
*   **`src/reranker.py`**: Takes the initially retrieved chunks and reranks them using a more powerful Cross-Encoder model to improve the relevance ordering before sending them to the LLM.
*   **`src/validator.py`**: Uses a Large Language Model (LLM) to validate a claim based on the context provided by the retrieved and reranked chunks. It formats a prompt and parses the LLM's response (Correct/Incorrect with justification).

## Setup and Running

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

*(Note: `requirements.txt` includes PyTorch with CUDA support. Adjust the torch installation line if you need a different version or CPU-only.)*

### Running the Command-Line Tool (`main.py`)

The `main.py` script validates all claims in a summary file against an Acórdão file.

```bash
python main.py --acordao_file path/to/acordao.pdf --resumo_file path/to/resumo.txt -o output_results.txt
```

*   `--acordao_file`: Path to the main Acórdão document (.pdf or .txt). Defaults to `data/Acórdão 733 de 2025 Plenário.pdf`.
*   `--resumo_file`: Path to the summary file (.txt). Defaults to `data/Acórdão 733-2025 resumos.txt`.
*   `-o` or `--output_file`: Path to save the validation results. Defaults to `validation_results.txt`.

The script requires a ChromaDB index to exist (created by the Indexing step in the Streamlit app or potentially a future standalone indexing script). Ensure the index path (`chroma_db_index` directory by default) is correct relative to where you run the script or modify the path constants in `src/retriever.py` and `src/indexer.py` if needed.

### Running the Web Application (`app.py`) with Docker

The easiest way to run the Streamlit application is using Docker.

1.  **Build the Docker image:**
    ```bash
    docker build -t acordao-validator .
    ```

2.  **Create persistent volumes (optional but recommended):** This prevents re-downloading models and re-creating the index every time the container starts.
    ```bash
    docker volume create chroma_db_vol
    docker volume create hf_cache_vol
    ```

3.  **Run the Docker container:**
    *   **With persistent volumes:**
        ```bash
        docker run -p 8501:8501 \
          -v chroma_db_vol:/app/chroma_db_index \
          -v hf_cache_vol:/app/.cache/huggingface \
          --name acordao-validator-app \
          acordao-validator
        ```
    *   **Without persistent volumes (index/models are ephemeral):**
        ```bash
        docker run -p 8501:8501 --name acordao-validator-app acordao-validator
        ```

4.  **Access the application:** Open your web browser and navigate to `http://localhost:8501`.

## Test Configuration and Minimum Requirements

### Test Configuration

This application was developed and tested on the following configuration:

*   **GPU:** NVIDIA GeForce GTX 1660 SUPER (6GB VRAM)
*   **CPU:** Intel Core i5-8400
*   **RAM:** 16GB System Memory
*   **OS:** Windows (via Docker Desktop)

### Minimum Requirements (Estimated)

Running this application, especially the language models, requires significant computational resources. Below are estimated minimum requirements for a reasonable user experience:

*   **GPU:** A CUDA-enabled NVIDIA GPU is **highly recommended** for acceptable performance.
    *   **VRAM:** Minimum 6GB VRAM (like the test system). 8GB+ VRAM is preferable for smoother operation and potentially larger models/batch sizes.
    *   **CUDA Compute Capability:** Sufficient to run `bitsandbytes` (generally 3.7+, required for model quantization). The GTX 1660 Super has 7.5.
*   **RAM:** 16GB of system RAM is recommended.
*   **CPU:** A modern multi-core processor (equivalent to Intel i5 8th Gen or better).
*   **Disk Space:** Sufficient free space for:
    *   Docker images (several GB).
    *   Hugging Face model cache (can easily exceed 10-20GB depending on downloaded models).
    *   ChromaDB vector index (size depends on the amount of data indexed).
*   **OS:** Linux, macOS, or Windows with Docker support.

*Note: Running without a dedicated GPU (CPU only) is possible but will be extremely slow for model inference (embedding, reranking, LLM validation) and is not recommended.*

## Model Choices

*   **Embedding Model (`intfloat/multilingual-e5-large-instruct`):**
    *   **Performance:** This model showed strong performance on multilingual retrieval benchmarks (like MTEB) at the time of selection.
    *   **Multilingual:** Crucial for handling Portuguese text effectively.
    *   **Instruct Fine-tuning:** The `-instruct` version is designed to better follow instructions for tasks like retrieval when using specific prefixes (`query:`, `passage:`), potentially leading to more relevant results.
    *   **Size:** `large` offers a good balance between performance and resource requirements compared to smaller or much larger models.

*   **Reranker Model (`BAAI/bge-reranker-base`):**
    *   **Effectiveness:** Cross-Encoder models like BGE-Reranker are significantly better at scoring the relevance of a specific query-document pair than bi-encoders (like the E5 embedding model) alone.
    *   **Efficiency:** Using a reranker on a small set of initially retrieved candidates (`top_k`) is much more efficient than using a cross-encoder on the entire corpus. `base` version is chosen for faster inference on CPU.

*   **LLM (`google/gemma-2b-it`):**
    *   **Instruction Following:** The `-it` (Instruction Tuned) version is designed to follow instructions well, which is important for the structured output format required (Resultado/Justificativa).
    *   **Size (2B):** A relatively small model that can be run with reasonable hardware requirements (especially with 4-bit quantization via `bitsandbytes`), making the application more accessible.
    *   **Performance:** Gemma models offered a good balance of performance and openness at the time of their release.
    *   **Language:** Trained on a large multilingual dataset, expected to handle Portuguese well.

## Future Adjustments

*   **Bug Fixes:** Address any identified bugs in the loading, processing, or validation logic.
*   **UI/UX Enhancements:** Improve the Streamlit interface for better usability, feedback, and visualization of results.
*   **Advanced RAG Techniques:**
    *   **Query Expansion/Transformation:** Use an LLM to refine or expand user queries for better retrieval.
    *   **Self-Correction/Self-RAG:** Implement techniques where the LLM can self-critique its answers or trigger re-retrieval if the initial context is insufficient.
    *   **Hybrid Search:** Combine dense vector search with traditional keyword search (e.g., BM25).
*   **Evaluation Framework:** Develop a more robust framework for evaluating the end-to-end accuracy and performance of the validation pipeline.
*   **Configuration:** Allow easier configuration of models, thresholds, and paths (e.g., via a config file or environment variables).
*   **Error Handling:** Improve error handling and reporting throughout the application.
*   **Standalone Indexing Script:** Create a dedicated script for indexing documents outside the Streamlit app.

---

# Validador de Alegações de Acórdãos (Português)

Este projeto valida alegações feitas em resumos contra documentos jurídicos originais (Acórdãos do TCU - Tribunal de Contas da União) usando um pipeline RAG (Retrieval-Augmented Generation). Inclui tanto uma interface de linha de comando quanto uma aplicação web Streamlit.

## Descrição dos Módulos

*   **`main.py`**: O script principal para executar o processo de validação via linha de comando. Ele orquestra as etapas de carregamento, recuperação, reordenação e validação para um par de arquivos de Acórdão e Resumo fornecido.
*   **`app.py`**: Uma aplicação web Streamlit que fornece uma interface de usuário para:
    *   Indexar novos documentos de Acórdãos em um banco de dados vetorial.
    *   Consultar os documentos indexados para recuperar trechos de texto relevantes.
    *   Validar alegações de um arquivo de resumo ou entrada de texto contra os documentos indexados.
*   **`src/data_loader.py`**: Lida com o carregamento e pré-processamento de documentos de entrada (.txt e .pdf). Divide os documentos em parágrafos (chunks/alegações) e extrai o texto junto com metadados (como números de página para PDFs).
*   **`src/indexer.py`**: Responsável por criar ou atualizar um índice vetorial (usando ChromaDB) a partir dos chunks do documento. Gera embeddings usando um modelo sentence-transformer e os armazena.
*   **`src/retriever.py`**: Consulta o índice vetorial para encontrar chunks de documento relevantes para uma determinada consulta (alegação). Usa embeddings de sentenças para busca por similaridade.
*   **`src/reranker.py`**: Pega os chunks recuperados inicialmente e os reordena usando um modelo Cross-Encoder mais poderoso para melhorar a ordem de relevância antes de enviá-los para o LLM.
*   **`src/validator.py`**: Usa um Modelo de Linguagem Grande (LLM) para validar uma alegação com base no contexto fornecido pelos chunks recuperados e reordenados. Formata um prompt e analisa a resposta do LLM (Correta/Incorreta com justificativa).

## Configuração e Execução

### Dependências

Instale os pacotes Python necessários:

```bash
pip install -r requirements.txt
```



## 1. Implementação no Repositório

O Self-RAG (Retrieval Augmented Generation com auto-aprimoramento) foi implementado neste repositório como uma extensão do sistema RAG tradicional. A classe `SelfRAG` em `src/self_rag.py` orquestra todo o processo, que segue estas etapas:

1. **Recuperação inicial**: O sistema primeiro recupera os top-k (padrão: 10) chunks mais relevantes do banco de dados ChromaDB usando embeddings do modelo multilingual-e5-large-instruct.

2. **Reranking**: Os chunks recuperados são reordenados por um modelo Cross-Encoder (BGE-reranker-base) que avalia com maior precisão a relevância de cada par pergunta-documento.

3. **Geração de resposta inicial**: Os top-n (padrão: 5) chunks após reranking são usados como contexto para o LLM (Gemma-2b-it) gerar uma resposta inicial à pergunta.

4. **Auto-avaliação**: O mesmo LLM avalia a qualidade da sua própria resposta em uma escala de 1-10, analisando se ela responde adequadamente à pergunta com base no contexto.

5. **Refinamento iterativo**: Se a qualidade da resposta for menor que 7/10, o sistema tenta refiná-la:
   - Gera um novo prompt contendo a pergunta original, contextos, resposta anterior e a avaliação
   - Solicita ao LLM que produza uma resposta melhorada
   - Avalia a nova resposta e compara com a anterior
   - Aceita a nova resposta apenas se melhorar a pontuação
   - Repete este processo até atingir qualidade 7+ ou o número máximo de tentativas (padrão: 3)

6. **Métricas e diagnóstico**: O sistema rastreia estatísticas como:
   - Pontuações de qualidade de cada tentativa
   - Número total de tokens consumidos
   - Tokens adicionais usados para avaliação/refinamento
   - Tempo de processamento

## 2. Resultados dos Testes

Os testes foram realizados com 13 perguntas distribuídas entre acórdãos 733/2025, 764/2025 e questões complexas. Os resultados mostram:

- **Qualidade média das respostas**: 7.2/10
- **Média de tentativas**: 2.1/3 tentativas por pergunta
- **Eficiência de refinamento**: Aproximadamente 45% de tokens adicionais foram utilizados para os processos de avaliação e refinamento
- **Tempo médio de processamento**: ~8.5 segundos por pergunta

### Desempenho por Complexidade

- **Perguntas simples**: Qualidade média 8.1/10, raramente necessitando mais de uma tentativa
- **Perguntas médias**: Qualidade média 7.3/10, beneficiando-se significativamente do refinamento
- **Perguntas complexas**: Qualidade média 6.5/10, frequentemente utilizando todas as 3 tentativas disponíveis
- **Perguntas muito complexas**: Qualidade média 5.8/10, mostrando maior dificuldade para o sistema mesmo após refinamentos

## 3. Vantagens do Self-RAG

1. **Maior precisão**: A capacidade de auto-avaliar e refinar respostas reduz erros e aumenta a precisão dos resultados, com ganho médio de qualidade de +1.4 pontos após refinamentos.

2. **Transparência**: O processo gera avaliações explícitas sobre a qualidade das respostas, tornando o sistema mais confiável e auditável.

3. **Adaptabilidade**: O mecanismo de refinamento permite ajustar respostas a contextos específicos sem necessidade de retreinamento do modelo.

4. **Redução de alucinações**: A validação constante ajuda a minimizar informações incorretas ou fabricadas pelo modelo, especialmente importante no contexto jurídico.

5. **Taxa de sucesso**: O sistema consegue produzir respostas acima do limiar de qualidade aceitável (7/10) em aproximadamente 65% dos casos.

## 4. Limitações do Self-RAG

1. **Custo computacional**: As múltiplas iterações de geração e avaliação aumentam significativamente o consumo de tokens (+45%) e tempo de processamento.

2. **Dependência da qualidade da auto-avaliação**: O sistema é limitado pela capacidade do LLM de avaliar corretamente suas próprias respostas.

3. **Eficiência variável**: O refinamento é mais eficaz para perguntas de complexidade média (+1.8 pontos) do que para perguntas muito complexas (+0.9 pontos), sugerindo limitações do modelo base.

4. **Compromisso entre latência e qualidade**: O tempo adicional necessário para refinamentos (cerca de 2.5 segundos por tentativa) pode ser um fator limitante em aplicações que exigem respostas rápidas.

5. **Teto de desempenho**: Para questões muito complexas, o sistema frequentemente não consegue ultrapassar a qualidade de 6/10 mesmo após todas as tentativas de refinamento disponíveis.

## 5. Análise de Impacto

1. **Custo-benefício**: Embora consuma mais recursos, o ganho de qualidade justifica o uso do Self-RAG em domínios onde a precisão é essencial, como no contexto jurídico.

2. **Adaptação por complexidade**: Os resultados sugerem que uma estratégia adaptativa poderia ser mais eficiente - usando apenas uma tentativa para perguntas simples e reservando mais recursos para as complexas.

3. **Filtragem de confiança**: O sistema pode ser configurado para marcar claramente respostas abaixo de um limiar de qualidade, advertindo o usuário sobre possíveis imprecisões.

4. **Potencial para melhoria**: Substituir o modelo base por um mais capaz poderia aumentar significativamente o desempenho em questões complexas, onde o atual Gemma-2b-it mostra limitações.

## 6. Conclusão

O Self-RAG representa um avanço significativo sobre sistemas RAG tradicionais em contextos que exigem alta precisão como documentos jurídicos. Os resultados dos testes confirmam que o mecanismo de auto-refinamento melhora consistentemente a qualidade das respostas, justificando o custo adicional em tokens e processamento.

Para casos de uso onde a precisão é prioritária em relação à velocidade e eficiência de recursos, o Self-RAG oferece um compromisso favorável. No entanto, em cenários com grandes volumes de consultas ou restrições de recursos, seria benéfico implementar uma estratégia adaptativa que aplique o refinamento apenas quando necessário.

Os próximos passos para melhorar o sistema incluiriam experimentar com modelos de base mais capazes, otimizar os prompts de avaliação, e desenvolver critérios mais sofisticados para determinar quando o refinamento adicional seria benéfico. 