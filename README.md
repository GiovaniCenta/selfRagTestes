# Acórdão Claim Validator

This project validates claims made in summaries against original legal documents (Acórdãos from TCU - Brazilian Court of Accounts) using a RAG (Retrieval-Augmented Generation) pipeline. It includes both a command-line interface and a Streamlit web application.

## Modules Description

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

*(Observação: `requirements.txt` inclui PyTorch com suporte a CUDA. Ajuste a linha de instalação do torch se precisar de uma versão diferente ou apenas para CPU.)*

### Executando a Ferramenta de Linha de Comando (`main.py`)

O script `main.py` valida todas as alegações em um arquivo de resumo contra um arquivo de Acórdão.

```bash
python main.py --acordao_file caminho/para/acordao.pdf --resumo_file caminho/para/resumo.txt -o resultados_validacao.txt
```

*   `--acordao_file`: Caminho para o documento principal do Acórdão (.pdf ou .txt). Padrão: `data/Acórdão 733 de 2025 Plenário.pdf`.
*   `--resumo_file`: Caminho para o arquivo de resumo (.txt). Padrão: `data/Acórdão 733-2025 resumos.txt`.
*   `-o` ou `--output_file`: Caminho para salvar os resultados da validação. Padrão: `validation_results.txt`.

O script requer que um índice ChromaDB exista (criado pela etapa de Indexação no aplicativo Streamlit ou, potencialmente, por um futuro script de indexação autônomo). Certifique-se de que o caminho do índice (diretório `chroma_db_index` por padrão) esteja correto em relação ao local de execução do script ou modifique as constantes de caminho em `src/retriever.py` e `src/indexer.py`, se necessário.

### Executando a Aplicação Web (`app.py`) com Docker

A maneira mais fácil de executar a aplicação Streamlit é usando Docker.

1.  **Construa a imagem Docker:**
    ```bash
    docker build -t acordao-validator .
    ```

2.  **Crie volumes persistentes (opcional, mas recomendado):** Isso evita baixar novamente os modelos e recriar o índice toda vez que o contêiner for iniciado.
    ```bash
    docker volume create chroma_db_vol
    docker volume create hf_cache_vol
    ```

3.  **Execute o contêiner Docker:**
    *   **Com volumes persistentes:**
        ```bash
        docker run -p 8501:8501 \
          -v chroma_db_vol:/app/chroma_db_index \
          -v hf_cache_vol:/app/.cache/huggingface \
          --name acordao-validator-app \
          acordao-validator
        ```
    *   **Sem volumes persistentes (índice/modelos são efêmeros):**
        ```bash
        docker run -p 8501:8501 --name acordao-validator-app acordao-validator
        ```

4.  **Acesse a aplicação:** Abra seu navegador e navegue para `http://localhost:8501`.

## Configuração de Teste e Requisitos Mínimos

### Configuração de Teste

Esta aplicação foi desenvolvida e testada na seguinte configuração:

*   **GPU:** NVIDIA GeForce GTX 1660 SUPER (6GB VRAM)
*   **CPU:** Intel Core i5-8400
*   **RAM:** 16GB de Memória do Sistema
*   **SO:** Windows (via Docker Desktop)

### Requisitos Mínimos (Estimados)

Executar esta aplicação, especialmente os modelos de linguagem, requer recursos computacionais significativos. Abaixo estão os requisitos mínimos estimados para uma experiência de usuário razoável:

*   **GPU:** Uma GPU NVIDIA habilitada para CUDA é **altamente recomendada** para desempenho aceitável.
    *   **VRAM:** Mínimo de 6GB VRAM (como o sistema de teste). 8GB+ VRAM é preferível para operação mais suave e modelos/tamanhos de lote potencialmente maiores.
    *   **Capacidade de Computação CUDA:** Suficiente para executar `bitsandbytes` (geralmente 3.7+, necessário para quantização do modelo). A GTX 1660 Super possui 7.5.
*   **RAM:** 16GB de RAM do sistema são recomendados.
*   **CPU:** Um processador multi-core moderno (equivalente a Intel i5 de 8ª geração ou superior).
*   **Espaço em Disco:** Espaço livre suficiente para:
    *   Imagens Docker (vários GB).
    *   Cache de modelos Hugging Face (pode facilmente exceder 10-20GB dependendo dos modelos baixados).
    *   Índice vetorial ChromaDB (tamanho depende da quantidade de dados indexados).
*   **SO:** Linux, macOS ou Windows com suporte a Docker.

*Observação: Executar sem uma GPU dedicada (apenas CPU) é possível, mas será extremamente lento para a inferência dos modelos (embedding, reranking, validação LLM) e não é recomendado.*

## Escolha dos Modelos

*   **Modelo de Embedding (`intfloat/multilingual-e5-large-instruct`):**
    *   **Desempenho:** Este modelo apresentou forte desempenho em benchmarks de recuperação multilíngue (como MTEB) no momento da seleção.
    *   **Multilíngue:** Crucial para lidar eficazmente com texto em português.
    *   **Ajuste Fino para Instruções:** A versão `-instruct` é projetada para seguir melhor as instruções para tarefas como recuperação ao usar prefixos específicos (`query:`, `passage:`), potencialmente levando a resultados mais relevantes.
    *   **Tamanho:** `large` oferece um bom equilíbrio entre desempenho e requisitos de recursos em comparação com modelos menores ou muito maiores.

*   **Modelo Reranker (`BAAI/bge-reranker-base`):**
    *   **Efetividade:** Modelos Cross-Encoder como o BGE-Reranker são significativamente melhores em pontuar a relevância de um par específico consulta-documento do que apenas bi-encoders (como o modelo de embedding E5).
    *   **Eficiência:** Usar um reranker em um pequeno conjunto de candidatos recuperados inicialmente (`top_k`) é muito mais eficiente do que usar um cross-encoder em todo o corpus. A versão `base` foi escolhida para inferência mais rápida na CPU.

*   **LLM (`google/gemma-2b-it`):**
    *   **Seguimento de Instruções:** A versão `-it` (Instruction Tuned) é projetada para seguir bem as instruções, o que é importante para o formato de saída estruturado necessário (Resultado/Justificativa).
    *   **Tamanho (2B):** Um modelo relativamente pequeno que pode ser executado com requisitos de hardware razoáveis (especialmente com quantização de 4 bits via `bitsandbytes`), tornando a aplicação mais acessível.
    *   **Desempenho:** Os modelos Gemma ofereciam um bom equilíbrio entre desempenho e abertura no momento de seu lançamento.
    *   **Idioma:** Treinado em um grande conjunto de dados multilíngue, espera-se que lide bem com o português.

## Ajustes Futuros

*   **Correções de Bugs:** Corrigir quaisquer bugs identificados na lógica de carregamento, processamento ou validação.
*   **Melhorias de UI/UX:** Aprimorar a interface Streamlit para melhor usabilidade, feedback e visualização dos resultados.
*   **Técnicas Avançadas de RAG:**
    *   **Expansão/Transformação de Consultas:** Usar um LLM para refinar ou expandir as consultas do usuário para melhor recuperação.
    *   **Autocorreção/Self-RAG:** Implementar técnicas onde o LLM pode autocriticar suas respostas ou acionar nova recuperação se o contexto inicial for insuficiente.
    *   **Busca Híbrida:** Combinar busca vetorial densa com busca tradicional por palavras-chave (ex: BM25).
*   **Framework de Avaliação:** Desenvolver um framework mais robusto para avaliar a precisão e o desempenho ponta a ponta do pipeline de validação.
*   **Configuração:** Permitir configuração mais fácil de modelos, limiares e caminhos (ex: via arquivo de configuração ou variáveis de ambiente).
*   **Tratamento de Erros:** Melhorar o tratamento e o relato de erros em toda a aplicação.
*   **Script de Indexação Autônomo:** Criar um script dedicado para indexar documentos fora do aplicativo Streamlit. 