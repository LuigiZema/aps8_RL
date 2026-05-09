# Relatório — APS 8: Coverage Path Planning

**Disciplina:** Reinforcement Learning — Insper  
**Autor:** Luigi Zema Matizonkas  
**Data:** 08/05/2026

---

## O problema

O agente precisa cobrir cada célula livre de um grid com obstáculos — sem nunca ter acesso ao mapa global. A única informação disponível a cada passo é a janela 3×3 ao redor do agente: quais células são livres, quais são obstáculos, qual é a cobertura atual. O ambiente é o `GridWorldCPPEnv` do professor, usado sem modificações.

A meta é atingir cobertura completa em ≥ 90 de 100 episódios tanto no 5×5 quanto no 10×10. Um ponto de bônus para quem conseguir o mesmo no 20×20.

---

## Por onde começa

Antes de escrever qualquer linha de código, rodei o baseline do professor para entender onde ele falha. O resultado foi 75/100 no 5×5 e 65/100 no 10×10 — útil como referência, mas claramente insuficiente.

O problema do baseline é estrutural: ele recebe o `coverage_ratio` (fração coberta), mas não **onde** estão as células não visitadas. O agente sabe que falta cobrir 30% do grid, mas não tem ideia de onde estão esses 30%. Sem essa informação, ele revisita células já visitadas, entra em loops, e frequentemente termina com uma ilha de células não visitadas no canto oposto ao seu ponto atual — sem conseguir chegar lá dentro do limite de passos.

---

## Primeira tentativa: observação 3×3 direta

A primeira ideia foi simples: usar a janela 3×3 diretamente, mas adicionar sinais calculados sobre um histórico de posições. Calculei direção e distância até a célula não visitada mais próxima usando um buffer de visitas recentes, e adicionei um histórico das 4 últimas ações para o agente detectar loops.

O resultado foi razoável no 5×5: 87/100. Mas no 10×10 zero-shot foi um desastre — **27/100**. O problema ficou claro: sem um mapa acumulado, o agente "esquece" o que visitou assim que sai do raio de visão 3×3. Em um grid maior, essa amnésia é fatal — o sinal de "célula mais próxima não visitada" aponta para uma célula que o agente já visitou há 50 passos, mas que o buffer não registrou.

---

## A virada: mapa acumulado + patch 5×5

A solução que funcionou foi construir um mapa interno persistente — um array `_mem` do mesmo tamanho do grid, atualizado a cada step com os 9 pixels da janela 3×3. Cada célula tem um estado: desconhecida (0), obstáculo (1), visitada (2), ou frontier — livre e vista, mas ainda não visitada (3).

Com esse mapa disponível, dois problemas se resolvem de uma vez:

**Primeiro:** em vez de passar a janela 3×3 bruta, extraio um **patch 5×5** centrado no agente diretamente do `_mem`. Isso dá ao agente um contexto espacial muito mais rico — ele vê 5 células em cada direção ao invés de 1 — e ainda é invariante ao tamanho do grid, porque o agente está sempre em `patch[2,2]`. O mesmo formato funciona no 5×5, no 10×10 e no 20×20.

**Segundo:** os sinais de frontier agora são calculados sobre o mapa inteiro, não só sobre a vizinhança imediata. O agente sabe a direção da célula não visitada mais próxima, sua distância, e quantas células não visitadas existem em cada quadrante. Isso funciona mesmo que a célula mais próxima esteja a 15 passos de distância.

Junto com isso, adicionei um **histórico das 4 últimas ações** como one-hot (16 bits), que permite ao modelo detectar o padrão de loop (direita→cima→esquerda→baixo repetido) e mudar de comportamento.

O resultado foi imediato: **97/100 no 5×5** após 600k steps. E o modelo treinado apenas no 5×5 atingiu **90/100 no 10×10 zero-shot** — sem nenhum retreinamento. A invariância do patch 5×5 funcionou exatamente como esperado.

Para o 10×10, continuar treinando a partir do modelo E1 com `set_env()` por mais 1.5M steps resultou em **91/100**. Os pontos 1 e 2 estavam garantidos.

---

## Tentativa de bônus: o 20×20

Com os dois pontos garantidos, começou a parte mais interessante — e mais frustrante: tentar atingir 90/100 no 20×20.

A primeira abordagem foi simplesmente continuar o currículo: carregar o modelo E2 e treinar no 20×20. Isso produziu ~78% nos primeiros experimentos, mas havia um problema: os primeiros testes usavam `max_steps=4000-5000`, muito acima do budget real do professor (1000 passos). Com `max_steps=1000`, o resultado ficou em **78/100 com mean_steps=664** — o agente usa bem o orçamento disponível, mas 22% dos episódios terminam em timeout.

### LSTM — o atalho que não funcionou

Em paralelo, tentei `RecurrentPPO + MultiInputLstmPolicy` na esperança de que a memória recorrente compensasse a necessidade de mapa acumulado. Não funcionou: o LSTM foi treinado com episódios de ~165 passos (10×10), e no 20×20 os episódios chegam a 1000 passos — 6× mais longos. O BPTT sobre sequências longas produziu gradientes que destruíram os pesos:

| | 10×10 antes | 10×10 depois | 20×20 |
|---|---|---|---|
| FrontierMemoryWrapper | 91% | — | 77% |
| RecurrentPPO + LSTM | — | **60%** | **44%** |

A lição foi importante: o mapa externo determinístico escala melhor que LSTM precisamente porque é construído sem gradiente. É exato, não sofre vanishing gradient, e funciona igualmente em qualquer grid.

### O diagnóstico do plateau

Depois de várias tentativas (progressive reward, estágio intermediário 15×15, potential-based shaping, currículo de obstáculos), o resultado continuava convergindo para 77-78%. Fiz uma análise de 200 episódios para entender por quê:

- **100% das falhas são truncamentos** — o agente não chega a uma conclusão errada, simplesmente fica sem tempo.
- **67% das falhas têm exatamente 1 célula faltando.** O agente cobre 399/400 células e fica sem passos.
- **Spawn não é a causa** — posição inicial em episódios que falham e que têm sucesso é estatisticamente idêntica.
- **As falhas são determinísticas** — as mesmas seeds sempre falham, nas mesmas coberturas.

O problema fica claro: certos layouts de obstáculos criam uma célula isolada que o agente não consegue alcançar dentro de 1000 passos. O agente **sabe** onde ela está — o sinal `dx_front/dy_front` aponta na direção certa — mas o caminho real até ela contorna vários obstáculos e é muito mais longo do que a distância Manhattan sugere.

### Por que o sinal de frontier engana

O sinal `dx_front/dy_front` calcula distância Manhattan — ele aponta para a célula mais próxima "em linha reta". Se há uma parede entre o agente e essa célula, o sinal diz "vá para a direita" quando o caminho real exige ir para a esquerda, dar a volta, e então entrar pela direita. O agente tenta ir direto, bate na parede, e fica oscilando — desperdiçando a maior parte dos 1000 passos disponíveis.

A correção lógica é usar **BFS a partir da posição atual**, respeitando os obstáculos já registrados em `_mem`, para calcular o caminho real até as células isoladas. O mapa acumulado já contém essa informação — falta apenas usá-la na hora de gerar o sinal de navegação.

### Diagnóstico de credit assignment (Deep-E3)

Antes da solução BFS, identifiquei e corrigi um problema de hiperparâmetros que limitava o treinamento no 20×20:

Com `n_steps=2048, N=8 envs` → apenas 256 passos/env entre updates. Um episódio 20×20 tem ~641 passos — ocupa 2.5 rollouts. O reward das últimas células sofre `γ^641 = 0.99^641 ≈ 0.001` — praticamente invisível ao gradiente.

A correção foi `n_steps=4096, N=4` (1024 passos/env) e `gamma=0.998` (peso no passo 641 = 0.28, 280× maior). Isso é o modelo "Deep-E3" — resultou em 77/100 estável, confirmando que o plateau não era de hiperparâmetros, mas estrutural da observação Manhattan.

---

## Resultados

| Modelo | 5×5 | 10×10 | 20×20 | max_steps |
|---|---|---|---|---|
| Baseline professor | 75/100 | 65/100 | — | 1000 |
| 3×3 direto (sem mapa) | 87/100 | 27/100 zero-shot | — | — |
| **E1 — mapa + patch 5×5** | **97/100 ✅** | 90/100 zero-shot | — | — |
| **E2 — continuação 10×10** | — | **91/100 ✅** | 77/100 zero-shot | — |
| LSTM (RecurrentPPO) | — | 60/100 | 44/100 | — |
| Deep-E3 (γ=0.998) | 95/100 | 91/100 | 77/100 | **1000** |
| **8.13 V3 — currículo obstáculos** | — | 89/100 | **78/100** | **1000** |

**Pontos garantidos: 2/2.** Melhor resultado bônus: 78/100 no 20×20 (arquivo `ppo_best_20x20.pt`).

---

## Os modelos entregues

| Arquivo | Formato | Descrição |
|---|---|---|
| `ppo_88_5x5.zip` | SB3 `.zip` | E1 — **97/100 no 5×5** (Ponto 1) |
| `ppo_88_10x10.zip` | SB3 `.zip` | E2 — **91/100 no 10×10** (Ponto 2) |
| `ppo_best_20x20.pt` | PyTorch `.pt` | Melhor bônus — 78/100 (8.13 V3) |

Os modelos `.zip` usam o formato nativo do SB3 (`model.save()`). O modelo 20×20 foi salvo como `.pt` porque o formato `.zip` do SB3 falha com PyTorch ≥ 2.1 (`RuntimeError: PytorchStreamReader`) — um bug de compatibilidade entre o leitor C++ legado e metadados adicionados em versões novas do PyTorch. O notebook inclui `bypass_load()` que contorna o problema para os `.zip`, e `load_pt()` para o `.pt`.

---

## O que fica de aprendizado

**Mapa externo explícito bate LSTM.** O FrontierMemoryWrapper é determinístico, exato, e escala para qualquer grid sem retreinamento. O LSTM precisa de BPTT — e em sequências longas isso destrói os pesos do curriculum.

**Observação invariante ao grid possibilita zero-shot.** O patch 5×5 centrado no agente é sempre o mesmo formato, independente do grid. Um modelo treinado no 5×5 funciona diretamente no 10×10 — e 90/100 zero-shot confirma que a representação generalizou.

**Credit assignment importa mais do que hiperparâmetros individuais.** A descoberta mais surpreendente foi que `n_steps=2048, N=8` efetivamente tornava o reward das últimas células invisível ao gradiente. Corrigir isso foi uma das mudanças mais impactantes.

**O plateau a 78% é estrutural.** Após todas as tentativas, ficou claro que a limitação não é de policy, reward, ou hiperparâmetros — é que o sinal Manhattan engana o agente nos layouts com células isoladas. A solução natural é BFS, e essa é a linha de desenvolvimento que ficou aberta.

---

## Referências

- Repositório do professor: [github.com/fbarth/gym_custom_env](https://github.com/fbarth/gym_custom_env)
- Raffin et al. (2021). *Stable-Baselines3: Reliable RL Implementations*. JMLR
- Schulman et al. (2017). *Proximal Policy Optimization*. arXiv:1707.06347
- Ng, Harada & Russell (1999). *Policy Invariance Under Reward Transformations*. ICML
- Dohare et al. (2024). *Loss of Plasticity in Deep Continual Learning*. Nature
