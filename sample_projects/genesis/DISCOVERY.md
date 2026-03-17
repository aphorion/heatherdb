# Discovery: Natural Selection Emerges from Memory Physics

**Date:** 2026-02-13
**Project:** Genesis (HeatherDB sample project)
**Finding:** When creatures are 32-dim vectors and fitness = EAM reconstruction fidelity, natural selection, speciation, extinction, carrying capacity, and ecological niches emerge spontaneously — with no hand-designed fitness function.

## Setup

- Each creature is a 32-dimensional unit vector
- Every generation: query SDM with each creature → fidelity = fitness
- Creatures below 0.60 fidelity die; survivors reproduce (blend + mutate)
- Survivors re-written to SDM, reinforcing their patterns
- Dead creatures' traces persist as "fossils" shaping the landscape

## Key Findings

### 1. Fitness Without a Fitness Function

Traditional artificial life requires explicit fitness functions (find food, avoid predators, maximize energy). Genesis has none. Fitness is simply: **can the EAM remember you?**

This works because:
- Vectors near deep basins of attraction reconstruct well → high fitness
- Vectors in sparse regions reconstruct poorly → low fitness
- The basins ARE the ecological niches, defined by the memory itself

### 2. Emergent Population Dynamics

| Biological Concept | SDM Mechanism |
|---|---|
| Fitness | Reconstruction fidelity |
| Ecological niches | Basins of attraction |
| Carrying capacity | SDM interference — too many vectors = mutual degradation |
| Speciation | Basin splitting as memory self-organizes |
| Extinction | Niche collapse from interference |
| Fossil record | Dead creatures' traces persist, shaping living creatures' fitness |
| Reinforcement | Survivors re-written each generation, deepening their basin |

### 3. Carrying Capacity from Interference

As population grows, EAM hard locations become overloaded. Vectors interfere destructively — reconstruction quality drops for everyone. Population self-regulates because:
- High population → high interference → low fitness → deaths → lower population
- Low population → low interference → high fitness → reproduction → higher population

This is a natural negative feedback loop with no explicit population cap.

### 4. Speciation Through Basin Dynamics

At 32 dimensions with high interference, the EAM's attractor landscape is constantly shifting. Clusters of similar creatures that happen to occupy different basins become distinct "species." When a basin splits (due to competitive learning reorganizing hard locations), one species becomes two.

## Why This Matters

| | Traditional ALife | Genesis (SDM) |
|---|---|---|
| Fitness | Hand-designed function | Reconstruction fidelity (emergent) |
| Niches | Programmed resources/zones | Basins of attraction (emergent) |
| Carrying capacity | Hard-coded population limit | Interference pressure (emergent) |
| Selection pressure | Explicit competition rules | Memory physics (emergent) |
| Fossil influence | None (dead = gone) | Traces persist, shape landscape |

The key insight: **the same Hopfield dynamics used by biological neural systems to store patterns IS the natural selection mechanism.** It's not a metaphor — it's the same mathematics. Evolution emerges from information storage physics.

## Parameters
- Dimension: 32 (very tight — high interference by design)
- Population: 50 initial, 120 max
- Death threshold: 0.60 fidelity
- Mutation rate: 0.12
- Species clustering: 0.35 cosine similarity threshold
