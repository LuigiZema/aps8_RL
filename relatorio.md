# Relatório — APS 8: Coverage Path Planning

**Disciplina:** Reinforcement Learning — Insper  
**Autor:** Luigi Zema Matizonkas  
**Data:** 08/05/2026  

## 1. Introdução

O objetivo desta APS foi desenvolver uma estratégia de aprendizado por reforço para resolver um problema de Coverage Path Planning. Nesse problema, o agente deve visitar todas as células livres de um grid com obstáculos.

O ambiente utilizado foi o `GridWorldCPPEnv`, disponibilizado pelo professor. O ambiente original não foi alterado diretamente. As modificações foram feitas por meio de wrappers e estratégias de treinamento.

A principal restrição do problema é que o agente não recebe o mapa global real. A observação original é apenas uma janela 3x3 ao redor da posição atual. Isso torna o problema mais difícil, porque o agente precisa decidir para onde ir sem saber diretamente quais regiões do grid ainda faltam ser visitadas.

A meta principal era atingir cobertura completa em pelo menos 90 de 100 episódios nos ambientes 5x5 e 10x10. O ambiente 20x20 foi utilizado como tentativa de bônus.

## 2. Baseline inicial

Antes de desenvolver a solução final, executei o baseline do professor para entender o comportamento inicial do agente.

Os resultados observados foram:

| Modelo | 5x5 | 10x10 |
|---|---:|---:|
| Baseline do professor | 75/100 | 65/100 |

Esse resultado mostrou que o PPO direto sobre a observação original consegue aprender parte da tarefa, mas ainda falha em muitos episódios.

A principal limitação observada foi que o agente sabe a fração de cobertura atingida, mas não sabe claramente onde estão as células ainda não visitadas. Com isso, ele tende a revisitar regiões já cobertas, entrar em loops e deixar pequenas áreas sem cobertura completa.

## 3. Representação de estado: 3x3 e 5x5

A observação original do ambiente é uma janela 3x3 ao redor do agente. Essa observação é importante porque representa a restrição principal do problema: o agente tem apenas informação local e não pode acessar diretamente o mapa global real.

Na primeira tentativa, mantive a representação mais próxima da observação original, usando a informação local 3x3 com alguns sinais adicionais. Essa abordagem melhorou o desempenho no 5x5, mas não generalizou bem para o 10x10.

Por isso, na solução final, usei uma representação baseada em um patch 5x5 centrado no agente. Esse patch 5x5 não é retirado do mapa global real. Ele é extraído da memória acumulada observada pelo agente, que é construída passo a passo a partir das observações locais recebidas durante o episódio.

A escolha pelo patch 5x5 teve três motivos principais:

- Dar mais contexto local ao agente do que a janela 3x3 original
- Reduzir loops e movimentos repetitivos em regiões já cobertas
- Melhorar a generalização do 5x5 para o 10x10, mantendo o mesmo formato de entrada para o modelo

Assim, a solução continua respeitando a limitação de observação parcial. O agente não recebe o mapa completo. Ele recebe uma representação local mais rica, construída apenas com informações que ele já observou.

## 4. Primeira tentativa: observação 3x3 com sinais adicionais

A primeira tentativa foi manter a observação 3x3, mas adicionar alguns sinais auxiliares ao agente, como direção aproximada até células ainda não visitadas e histórico recente de ações.

Essa abordagem melhorou o desempenho no 5x5, mas não generalizou bem para o 10x10.

| Estratégia | 5x5 | 10x10 zero-shot |
|---|---:|---:|
| Frontier 3x3 sem mapa acumulado | 87/100 | 27/100 |

O resultado no 10x10 mostrou que a observação 3x3 ainda era limitada demais. Em grids maiores, o agente rapidamente perde referência de regiões já visitadas e de regiões que ainda faltam ser cobertas.

Essa tentativa foi importante para mostrar que o problema não era apenas de algoritmo ou hiperparâmetro. A representação do estado precisava carregar melhor a informação espacial acumulada ao longo do episódio.

## 5. Solução final: FrontierMemoryWrapper

A solução que apresentou melhor desempenho foi a criação do `FrontierMemoryWrapper`.

Esse wrapper mantém uma memória acumulada chamada `_mem`. Essa memória não representa o mapa global real do ambiente. Ela é construída passo a passo, usando apenas a janela 3x3 observada pelo agente em cada instante.

Cada célula da memória pode representar, de forma simplificada:

- célula desconhecida
- obstáculo observado
- célula livre já visitada
- célula livre observada, mas ainda não visitada

A partir dessa memória acumulada, o wrapper gera uma observação mais rica para o modelo PPO.

## 6. Componentes da representação

A observação final usada pelo agente combina três grupos principais de informação.

### 6.1 Patch 5x5 centrado no agente

Em vez de usar apenas a janela 3x3 original, passei a usar um patch 5x5 extraído da memória acumulada.

O agente fica sempre no centro desse patch. Isso torna a representação invariante ao tamanho do grid. O mesmo formato de entrada pode ser usado em ambientes 5x5, 10x10 e 20x20.

Essa escolha foi importante para permitir generalização entre tamanhos diferentes de ambiente.

### 6.2 Sinais de frontier

Além do patch 5x5, o wrapper calcula sinais relacionados às células ainda não visitadas, chamadas de frontier.

Esses sinais incluem:

- direção aproximada até a célula frontier mais próxima
- distância aproximada até essa célula
- quantidade de células frontier por quadrante

Essas informações ajudam o agente a entender para qual região do grid ainda existe cobertura pendente.

### 6.3 Histórico de ações

Também foi adicionado um histórico das últimas 4 ações, codificado como one-hot.

Esse histórico ajuda o agente a identificar padrões de loop. Por exemplo, quando ele começa a repetir movimentos como direita, cima, esquerda e baixo, essa informação aparece no estado e pode ser usada pela política para evitar ciclos improdutivos.

## 7. Treinamento com PPO

O algoritmo principal utilizado foi o PPO, por ser estável e adequado para ambientes com política estocástica.

A estratégia de treinamento foi organizada em etapas:

1. Treinamento inicial no ambiente 5x5
2. Avaliação zero-shot no 10x10
3. Continuação do treinamento no ambiente 10x10
4. Tentativas adicionais no ambiente 20x20 para o bônus

O modelo treinado no 5x5 conseguiu generalizar diretamente para o 10x10, o que confirmou que a representação baseada em memória acumulada e patch centrado no agente estava funcionando.

## 8. Resultados principais

Os principais resultados foram:

| Experimento | 5x5 | 10x10 | 20x20 | Observação |
|---|---:|---:|---:|---|
| Baseline do professor | 75/100 | 65/100 | — | Referência inicial |
| Frontier 3x3 sem mapa acumulado | 87/100 | 27/100 zero-shot | — | Não generalizou para 10x10 |
| E1 — PPO + FrontierMemoryWrapper | 97/100 | 90/100 zero-shot | — | Treinado no 5x5 |
| E2 — Curriculum 5x5 para 10x10 | — | 91/100 | 77/100 zero-shot | Modelo final do critério principal |
| LSTM | — | 60/100 | 44/100 | Tentativa com memória recorrente |
| Melhor tentativa 20x20 | — | 89/100 | 78/100 | Melhor resultado de bônus |

Os modelos finais entregues foram:

| Arquivo | Descrição |
|---|---|
| `ppo_88_5x5.zip` | Modelo PPO com 97/100 no 5x5 |
| `ppo_88_10x10.zip` | Modelo PPO com 91/100 no 10x10 |
| `ppo_best_20x20.pt` | Melhor modelo obtido para o 20x20, com 78/100 |

## 9. Análise do desempenho no 5x5 e 10x10

No 5x5, o modelo final atingiu 97/100. Esse resultado mostra que a política aprendeu uma estratégia de cobertura bastante robusta para o ambiente menor.

O ponto mais relevante, porém, foi o resultado zero-shot no 10x10. O modelo treinado apenas no 5x5 atingiu 90/100 no 10x10, antes de qualquer retreinamento nesse tamanho.

Isso indica que a representação proposta não estava simplesmente memorizando trajetórias do 5x5. Ela permitiu ao agente transferir uma estratégia de cobertura para um ambiente maior.

Depois da continuação do treinamento no 10x10, o desempenho final chegou a 91/100, atendendo ao critério principal da APS.

## 10. Tentativa de bônus no 20x20

Após atingir os resultados necessários no 5x5 e no 10x10, testei diferentes estratégias para melhorar o desempenho no 20x20.

Entre as tentativas, foram explorados:

- continuação do curriculum para 20x20
- ajuste de hiperparâmetros
- uso de estágios intermediários
- reward shaping
- tentativa com LSTM
- aumento de horizonte de rollout
- análise do limite de passos

O melhor resultado obtido foi 78/100 no 20x20.

Esse resultado ficou abaixo do critério de bônus, mas mostrou que a estratégia ainda conseguia cobrir quase todo o grid em muitos episódios.

## 11. Por que o 20x20 travou em aproximadamente 78%

A análise dos episódios com falha indicou que o problema principal não era a falta de cobertura média. Em muitos casos, o agente chegava muito próximo da cobertura completa, mas deixava uma ou poucas células faltando.

O principal motivo identificado foi a limitação do sinal de frontier baseado em distância aproximada. Esse sinal indica a direção de uma célula ainda não visitada, mas não considera necessariamente o caminho real até ela quando existem obstáculos no meio.

Em alguns layouts, a célula restante parece próxima pela distância Manhattan, mas o caminho real exige dar uma volta longa em torno de obstáculos. Nesses casos, o agente pode tentar ir na direção aparentemente correta, bater em obstáculos ou oscilar, gastando muitos passos.

Por isso, a limitação do 20x20 parece estar mais ligada à qualidade do sinal de navegação do que à capacidade básica do PPO.

Uma melhoria natural seria substituir a direção baseada em distância Manhattan por uma busca BFS sobre a memória acumulada observada. Isso permitiria calcular caminhos mais realistas até as células frontier, respeitando os obstáculos já conhecidos pelo agente.

## 12. Tentativa com LSTM

Também testei uma abordagem com memória recorrente usando LSTM.

A hipótese era que a LSTM poderia aprender uma memória interna útil para lidar com a observação parcial. Porém, o resultado foi inferior ao wrapper com memória externa.

| Modelo | 10x10 | 20x20 |
|---|---:|---:|
| FrontierMemoryWrapper | 91/100 | 77/100 zero-shot |
| RecurrentPPO com LSTM | 60/100 | 44/100 |

A principal conclusão foi que a memória externa determinística funcionou melhor do que tentar aprender memória apenas por recorrência.

No caso do LSTM, a sequência de decisões no 20x20 é muito mais longa. Isso dificulta o treinamento por BPTT e pode causar perda de desempenho em relação aos modelos anteriores.

## 13. Principais aprendizados

O primeiro aprendizado foi que a representação do estado é decisiva em problemas de observação parcial. O PPO sozinho, usando apenas a janela 3x3, não tinha informação suficiente para resolver bem o problema em grids maiores.

O segundo aprendizado foi que uma memória externa simples pode ser mais eficiente do que uma memória recorrente aprendida. O `FrontierMemoryWrapper` guarda de forma explícita o que já foi observado, sem depender de gradientes para lembrar regiões antigas do mapa.

O terceiro aprendizado foi que a invariância da observação ajudou muito na generalização. Como o agente sempre recebe um patch 5x5 centrado nele, a entrada do modelo tem o mesmo formato no 5x5 e no 10x10.

Por fim, o 20x20 mostrou que a cobertura completa em ambientes maiores exige não apenas memória, mas também um sinal de navegação mais inteligente. A próxima melhoria natural seria calcular caminhos até frontier usando BFS sobre a memória acumulada observada.

## 14. Conclusão

A solução desenvolvida atingiu o objetivo principal da APS.

O modelo final obteve:

- 97/100 no ambiente 5x5
- 91/100 no ambiente 10x10
- 78/100 no ambiente 20x20, como tentativa de bônus

A estratégia mais importante foi a criação do `FrontierMemoryWrapper`, que transformou a observação parcial em uma representação mais útil, mantendo a restrição de não acessar o mapa global real do ambiente.

A solução demonstrou boa generalização do 5x5 para o 10x10 e permitiu cumprir os critérios principais da atividade.

## 15. Referências

- Material da aula 23: https://insper.github.io/rl/classes/23_custom_env_agent/
- Repositório do professor: https://github.com/fbarth/gym_custom_env
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
- Raffin et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations.
