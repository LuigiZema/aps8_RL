# APS 8 — Coverage Path Planning com PPO e Frontier Memory

**Disciplina:** Reinforcement Learning — Insper  
**Autor:** Luigi Zema Matizonkas  
**Data:** 08/05/2026  

## 1. Descrição do problema

Nesta APS, o objetivo foi treinar um agente capaz de visitar todas as células livres de um grid com obstáculos. O ambiente utilizado foi o `GridWorldCPPEnv`, disponibilizado no repositório do professor, sem alteração direta no arquivo do ambiente.

A principal dificuldade do problema é que o agente não recebe o mapa global real. A observação original disponível a cada passo é uma janela 3x3 ao redor da posição atual. Portanto, o agente precisa tomar decisões com informação parcial e aprender uma estratégia de cobertura eficiente.

O critério principal da atividade é obter cobertura completa em pelo menos 90 de 100 episódios nos ambientes 5x5 e 10x10. O ambiente 20x20 foi tratado como tentativa de bônus.

## 2. Estratégia proposta

A solução desenvolvida foi baseada em PPO, usando a biblioteca Stable-Baselines3, combinado com um wrapper próprio chamado `FrontierMemoryWrapper`.

A ideia central do wrapper é manter uma memória acumulada das células já observadas pelo agente. Essa memória não é o mapa global real do ambiente. Ela é construída progressivamente, apenas a partir das observações locais que o agente recebe durante o episódio.

A observação original do ambiente é 3x3. Porém, na solução final, usei uma representação de estado baseada em um patch 5x5 centrado no agente. Esse patch 5x5 não vem diretamente do mapa completo. Ele é extraído da memória acumulada observada, ou seja, contém apenas informações que o agente já teve oportunidade de observar.

A escolha pelo patch 5x5 foi feita porque a janela 3x3 ficou limitada para generalizar do 5x5 para o 10x10. O patch 5x5 dá mais contexto local ao agente, ajuda a reduzir loops e melhora a identificação de regiões próximas ainda não cobertas, sem violar a restrição de observação parcial.

Com essa memória observada, o wrapper gera uma representação de estado mais informativa para o PPO:

- Patch 5x5 centrado no agente, extraído da memória acumulada observada
- Sinais de frontier, indicando direção e distância aproximada até células livres ainda não visitadas
- Contagem de células frontier por quadrante
- Histórico das últimas 4 ações, para ajudar o agente a reduzir loops

Essa representação mantém a observação compatível com o princípio de informação parcial, pois o agente não recebe o mapa global real do ambiente. Ele usa apenas o que já foi observado ao longo da trajetória.

## 3. Principais resultados

| Experimento | 5x5 | 10x10 | 20x20 | Observação |
|---|---:|---:|---:|---|
| Baseline do professor | 75/100 | 65/100 | — | Referência inicial |
| Frontier 3x3 sem mapa acumulado | 87/100 | 27/100 zero-shot | — | Não generalizou para 10x10 |
| E1 — PPO + FrontierMemoryWrapper | 97/100 | 90/100 zero-shot | — | Modelo treinado no 5x5 |
| E2 — Curriculum 5x5 para 10x10 | — | 91/100 | 77/100 zero-shot | Modelo final para o 10x10 |
| Melhor tentativa 20x20 | — | 89/100 | 78/100 | Tentativa de bônus |

Os critérios principais da APS foram atendidos nos ambientes 5x5 e 10x10.

O bônus 20x20 foi explorado, mas o melhor resultado obtido foi 78/100, abaixo do critério de 90/100.

## 4. Arquivos do repositório

| Arquivo | Descrição |
|---|---|
| `aps8_final.ipynb` | Notebook principal com implementação, treino, avaliação e análise dos experimentos |
| `relatorio.md` | Relatório técnico explicando a estratégia, os resultados e as limitações |
| `grid_world_cpp.py` | Ambiente original do professor, sem modificações diretas |
| `evaluation_results_final.json` | Arquivo com os resultados finais dos experimentos |
| `ppo_88_5x5.zip` | Modelo PPO treinado no 5x5, com 97/100 |
| `ppo_88_10x10.zip` | Modelo PPO ajustado para o 10x10, com 91/100 |
| `ppo_best_20x20.pt` | Melhor modelo obtido na tentativa de bônus 20x20, com 78/100 |
| `curve_frontier_5x5.png` | Curva de treinamento do modelo 5x5 |
| `curve_frontier_10x10.png` | Curva de treinamento do modelo 10x10 |
| `curve_82_E3.png` | Curva de uma das tentativas no 20x20 |
| `coverage_comparison_88.png` | Comparação dos resultados principais |
| `comparison_82.png` | Comparação das tentativas no 20x20 |

## 5. Como executar

Instale as dependências principais:

```bash
pip install "gymnasium>=1.0" "stable-baselines3>=2.0" "torch>=2.0" numpy matplotlib pandas seaborn jupyter
```

Abra o notebook:

```bash
jupyter notebook aps8_final.ipynb
```

O notebook foi estruturado para detectar modelos já salvos e evitar retreinamento desnecessário. Caso os arquivos dos modelos estejam presentes, a avaliação pode ser executada diretamente.

Para forçar novo treinamento, basta remover os arquivos:

```bash
ppo_88_5x5.zip
ppo_88_10x10.zip
ppo_best_20x20.pt
```

## 6. Observações importantes

O wrapper desenvolvido não fornece ao agente o mapa global real do ambiente. A memória `_mem` representa somente o conjunto de células que foram observadas pelo agente durante o episódio.

Essa distinção é importante porque a estratégia respeita a limitação de observação parcial do problema, mas adiciona uma memória externa construída de forma incremental, algo necessário para lidar com tarefas de cobertura.

Também é importante destacar a diferença entre a observação original e a representação usada no modelo final:

| Elemento | Explicação |
|---|---|
| Observação original 3x3 | Janela local fornecida pelo ambiente ao redor do agente |
| Memória acumulada observada | Registro das células que o agente já viu durante o episódio |
| Patch 5x5 final | Recorte local extraído da memória acumulada observada, centrado no agente |

Portanto, o patch 5x5 não significa acesso ao mapa completo. Ele apenas organiza melhor a informação parcial já observada pelo agente.

## 7. Referências

- Material da aula 23: https://insper.github.io/rl/classes/23_custom_env_agent/
- Repositório do professor: https://github.com/fbarth/gym_custom_env
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
- Raffin et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations.
