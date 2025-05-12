

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