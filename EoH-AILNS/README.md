# EoH-ALNS: Evolution of Heuristics for Adaptive Large Neighborhood Search

An Evolution of Heuristics framework for automatically generating and evolving ALNS operators using Large Language Models for the Permutation Flow Shop Problem (PFSP) and Prize-Collecting Traveling Salesman Problem (PCTSP).

## Overview

This project implements an EoH-ALNS framework that:
- Generates ALNS operators (destroy and repair) using LLMs
- Evolves operators through multiple evolution strategies
- Supports both PFSP and PCTSP optimization problems
- Evaluates and stores results in JSON format

## Quick Start

### PFSP Implementation
```bash
python demo_simple_eoh.py          # Generate and evolve PFSP operators 
python eoh_evolution_pfsp.py        # Generate and evolve PFSP operators extensively
python quick_evaluation.py         # Evaluate best PFSP operator
```

### PCTSP Implementation
```bash
python demo_eoh_pctsp.py           # Generate and evolve PCTSP operators
python eoh_evolution_pctsp.py        # Generate and evolve PCTSP operators extensively
python evaluate_pctsp_operators.py # Evaluate PCTSP operators
```

## Core Files

### PFSP Implementation
- `eoh_pfsp.py` - Main EoH framework for PFSP
- `eoh_evolution_pfsp.py` - Evolution strategies for PFSP
- `PFSP.py` - Core PFSP functions and classes
- `pfsp_prompts.py` - LLM prompts for PFSP operators

### PCTSP Implementation
- `eoh_pctsp.py` - Main EoH framework for PCTSP
- `eoh_evolution_pctsp.py` - Evolution strategies for PCTSP
- `PCTSP.py` - Core PCTSP functions and classes
- `pctsp_prompts.py` - LLM prompts for PCTSP operators

### Shared Components
- `interface_llm.py` - LLM wrapper interface
- `llm_api.py` - Core LLM interface
- `data/` - Benchmark instances for both problems

## Usage

### Basic PFSP Usage
```python
from eoh_pfsp import EoH_PFSP

eoh = EoH_PFSP(
    pop_size=4,
    n_generations=3,
    data_file="data/j50_m20/j50_m20_01.txt"
)

final_population = eoh.run()
best = final_population[0]
```

### Basic PCTSP Usage
```python
from eoh_pctsp import EoH_PCTSP

eoh = EoH_PCTSP(
    pop_size=4,
    n_generations=3,
    data_file="data/pctsp_instances/instance01.txt"
)

final_population = eoh.run()
best = final_population[0]
```

## Evolution Strategies

Both implementations support the following evolution strategies:
- **i1**: Generate initial operators from scratch
- **e1**: Create different algorithms (exploration)
- **e2**: Combine and improve existing operators (exploitation)
- **m1**: Modify existing algorithms significantly
- **m2**: Simplify for better generalization

## Results Structure

```
demo_results/
├── generations/
│   ├── generation_0.json
│   ├── generation_1.json
│   └── ...
└── final_summary.json
```

Each generated operator includes complete code, performance metrics, and metadata. 