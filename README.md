# APS 8 — Coverage Path Planning com PPO + Frontier Memory

**Disciplina:** Reinforcement Learning — Insper  
**Autor:** Luigi Zema Matizonkas  
**Data:** 08/05/2026

---

## O problema

O agente precisa visitar todas as células livres de um grid com obstáculos. A única observação disponível é a janela 3×3 imediatamente ao redor — sem mapa global, sem GPS, sem memória embutida no ambiente. O algoritmo de base (PPO direto sobre a observação 3×3) atinge 75/100 no 5×5 e 65/100 no 10×10.

---

## A solução: FrontierMemoryWrapper

A ideia central é um **wrapper Gymnasium** que mantém um mapa interno acumulado (`_mem`) e expõe ao modelo uma representação muito mais rica do que a observação bruta 3×3:

- **Patch 5×5** extraído do `_mem`, com o agente sempre no centro — invariante ao tamanho do grid
- **Sinais de frontier** calculados sobre o mapa inteiro: direção e distância até célula não visitada, contagem por quadrante
- **Histórico de ações** (últimas 4, como one-hot de 16 bits) para detectar loops

O resultado-chave da invariância: um modelo treinado apenas no 5×5 consegue **90/100 no 10×10 zero-shot** — sem retreinamento.

---

## Resultados

| Modelo | 5×5 | 10×10 | 20×20 | Status |
|---|---|---|---|---|
| Baseline professor | 75/100 | 65/100 | — | referência |
| **E1 — 5×5 (600k steps)** | **97/100** | 90/100 zero-shot | — | ✅ **Ponto 1** |
| **E2 — 10×10 (1.5M steps)** | — | **91/100** | 77/100 zero-shot | ✅ **Ponto 2** |
| Melhor bônus (8.13 V3) | — | 89/100 | **78/100** | bônus parcial |

**Pontuação garantida: 2/2.**

---

## Arquivos

| Arquivo | Descrição |
|---|---|
| `aps8_final.ipynb` | Notebook completo — treino E1, E2, avaliações, curvas, bônus 20×20 |
| `relatorio.md` | Relatório com narrativa do desenvolvimento e análise técnica |
| `grid_world_cpp.py` | Ambiente do professor — sem modificações |
| `ppo_88_5x5.zip` | Modelo E1 — **97/100 no 5×5** (Ponto 1) |
| `ppo_88_10x10.zip` | Modelo E2 — **91/100 no 10×10** (Ponto 2) |
| `ppo_best_20x20.pt` | Melhor bônus — 78/100 no 20×20 |
| `curve_frontier_5x5.png` | Curva de treino E1 |
| `curve_frontier_10x10.png` | Curva de treino E2 |
| `curve_82_E3.png` | Curva Deep-E3 (bônus) |
| `coverage_comparison_88.png` | Comparação E1/E2 vs baseline |
| `comparison_82.png` | Comparação todas as tentativas 20×20 |
| `evaluation_results_final.json` | Métricas completas por experimento |

---

## Como executar

```bash
pip install gymnasium>=1.0 stable-baselines3>=2.0 torch>=2.0 numpy matplotlib pandas seaborn
jupyter notebook aps8_final.ipynb
```

O notebook detecta automaticamente se os modelos já existem (`SKIP_IF_EXISTS=True`) e carrega sem retreinar (~5 segundos). Para forçar retreinamento completo, delete `ppo_88_5x5.zip` e `ppo_88_10x10.zip`.

Tempo estimado de treinamento do zero: ~28 minutos (E1: ~4 min, E2: ~24 min).

---

## Referências

- Repositório do professor: [github.com/fbarth/gym_custom_env](https://github.com/fbarth/gym_custom_env)
- Material da aula 23: [insper.github.io/rl/classes/23_custom_env_agent](https://insper.github.io/rl/classes/23_custom_env_agent/)
- Raffin et al. (2021). *Stable-Baselines3*. JMLR
- Schulman et al. (2017). *Proximal Policy Optimization*. arXiv:1707.06347
- Ng, Harada & Russell (1999). *Policy Invariance Under Reward Transformations*. ICML
- Dohare et al. (2024). *Loss of Plasticity in Deep Continual Learning*. Nature
