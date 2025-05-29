# EoH-PCTSP: Evolution of Heuristics for Price Collecting TSP

This directory contains the **Evolution of Heuristics (EoH) framework** adapted for the **Price Collecting Travelling Salesman Problem (PCTSP)**. The framework uses Large Language Models (LLMs) to automatically evolve repair operators for the ALNS (Adaptive Large Neighborhood Search) metaheuristic.

## 🎯 Problem Description

The **Price Collecting Travelling Salesman Problem (PCTSP)** is a variant of the TSP where:
- A salesman starts and ends at a depot
- Each node has an associated **prize** (reward for visiting) and **penalty** (cost for skipping)
- The goal is to find a tour that **minimizes total cost** = tour length + penalties for unvisited nodes
- The tour must collect at least a **minimum required prize** (constraint)

## 🧬 EoH Framework Overview

The EoH framework automatically evolves PCTSP repair operators through:

1. **Initialization (i1)**: Generate initial population of repair operators using LLM
2. **Evolution Strategies**:
   - **e1**: Create totally different algorithms inspired by parents
   - **e2**: Combine and improve existing operators
   - **m1**: Modify existing algorithms
   - **m3**: Simplify operators for generalization
3. **Evaluation**: Test operators using ALNS on PCTSP instances
4. **Selection**: Keep best operators for next generation

## 📁 File Structure

```
PCTSP/
├── PCTSP.py                    # Core PCTSP classes and functions
├── pctsp_prompts.py           # PCTSP-specific LLM prompts
├── eoh_evolution_pctsp.py     # Evolution strategies for PCTSP
├── eoh_pctsp.py               # Main EoH-PCTSP framework
├── run_eoh_pctsp.py           # Command-line execution script
├── demo_eoh_pctsp.py          # Demo without LLM API
├── test_eoh_pctsp.py          # Test suite
├── README_EoH_PCTSP.md        # This file
└── data/                      # PCTSP instance files
    └── pctsp_20_20_instances.pkl
```

## 🚀 Quick Start

### 1. Run Demo (No LLM Required)

Test the framework without needing an LLM API:

```bash
cd PCTSP
python demo_eoh_pctsp.py
```

### 2. Run Tests

Verify the framework works correctly:

```bash
cd PCTSP
python test_eoh_pctsp.py
```

### 3. Run Full EoH Framework

With LLM API (requires API endpoint and key):

```bash
cd PCTSP
python run_eoh_pctsp.py --api_endpoint YOUR_API_URL --api_key YOUR_API_KEY
```

## 🔧 Usage Examples

### Basic Usage

```bash
# Run with default parameters (4 individuals, 3 generations, size 20)
python run_eoh_pctsp.py

# Custom parameters
python run_eoh_pctsp.py --pop_size 6 --generations 5 --problem_size 20

# Use only 1 instance for faster testing
python run_eoh_pctsp.py --max_instances 1

# Custom output directory
python run_eoh_pctsp.py --output_dir my_results
```

### With LLM API

```bash
# Google Gemini API
python run_eoh_pctsp.py \
    --api_endpoint "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent" \
    --api_key "YOUR_GEMINI_API_KEY" \
    --model_llm "gemini-pro"

# OpenAI API
python run_eoh_pctsp.py \
    --api_endpoint "https://api.openai.com/v1/chat/completions" \
    --api_key "YOUR_OPENAI_API_KEY" \
    --model_llm "gpt-4"
```

## 📊 Output Structure

The framework generates results in the specified output directory:

```
eoh_pctsp_results/
├── generations/
│   ├── generation_0.json      # Initial population
│   ├── generation_1.json      # Generation 1 results
│   └── ...
├── best/
│   ├── best_gen_0.json        # Best individual per generation
│   ├── best_gen_1.json
│   └── ...
└── final_summary.json         # Complete run summary
```

### Result Format

Each individual contains:
```json
{
  "algorithm": "Description of the algorithm",
  "code": "def llm_repair(state, rng, **kwargs): ...",
  "objective": 45.67,
  "gap": -12.34,
  "runtime": 0.123,
  "feasible": true,
  "tour_length": 8,
  "prize_collected": 1.25,
  "strategy": "e1",
  "generation": 1,
  "timestamp": "2024-01-01 12:00:00"
}
```

## 🧪 Core Components

### PCTSP Classes

- **`PCTSPData`**: Represents a PCTSP instance with nodes, depot, prizes, and penalties
- **`PCTSPSolution`**: Represents a solution with tour and unvisited nodes
- **`construct_initial_solution()`**: Creates feasible initial solutions

### Operators

- **Destroy Operators**: `random_removal`, `worst_removal`
- **Repair Operators**: `greedy_repair` (baseline), LLM-generated operators
- **Evaluation**: `evaluate_operator()` using ALNS

### Evolution Engine

- **`Evolution`**: Implements evolution strategies (i1, e1, e2, m1, m3)
- **`EoH_PCTSP`**: Main framework orchestrating the evolution process

## 🎛️ Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pop_size` | 4 | Population size |
| `--generations` | 3 | Number of generations |
| `--problem_size` | 20 | Problem size (20, 50, 100) |
| `--max_instances` | 2 | Max instances to use |
| `--output_dir` | `eoh_pctsp_results` | Output directory |
| `--api_endpoint` | None | LLM API endpoint |
| `--api_key` | None | LLM API key |
| `--model_llm` | None | LLM model name |
| `--debug` | False | Enable debug mode |

### Framework Parameters

```python
eoh = EoH_PCTSP(
    pop_size=4,           # Population size
    n_generations=3,      # Number of generations
    problem_size=20,      # Problem size (nodes)
    max_instances=2,      # Max instances to evaluate on
    output_dir="results", # Output directory
    debug_mode=False      # Debug mode
)
```

## 🔍 PCTSP-Specific Features

### Problem Encoding

- **Tour**: List of visited nodes (excluding depot)
- **Unvisited**: List of nodes not in tour (incur penalties)
- **Objective**: Tour length + sum of penalties for unvisited nodes
- **Constraint**: Total prize collected ≥ required prize (usually 1.0)

### Key Considerations

1. **Prize-Penalty Trade-off**: Balance visiting nodes (tour cost) vs. skipping them (penalty)
2. **Feasibility**: Must collect enough prize to satisfy constraint
3. **Distance**: Euclidean distance between nodes and depot
4. **Insertion**: Smart positioning of nodes in tour to minimize cost

### Repair Operator Template

```python
def llm_repair(state, rng, **kwargs):
    # Access problem data
    # DATA.size, DATA.depot, DATA.locations, DATA.prizes, DATA.penalties
    
    # Process unvisited nodes
    for node in state.unvisited:
        # Calculate metrics (distance, prize/penalty ratio, etc.)
        # Decide whether to insert node
        if should_insert(node):
            state.opt_insert(node)  # Insert at best position
    
    # Ensure feasibility
    while not state.is_feasible() and state.unvisited:
        node = select_best_node(state.unvisited)
        state.opt_insert(node)
    
    return state
```

## 📈 Performance Metrics

- **Objective Value**: Total cost (tour length + penalties)
- **Gap**: Percentage improvement over initial solution
- **Feasibility**: Whether solution satisfies prize constraint
- **Tour Length**: Number of nodes visited
- **Prize Collected**: Total prize from visited nodes
- **Runtime**: Execution time for evaluation

## 🔧 Troubleshooting

### Common Issues

1. **No instances found**: Check that `data/pctsp_20_20_instances.pkl` exists
2. **LLM API errors**: Verify API endpoint and key are correct
3. **Import errors**: Ensure all dependencies are installed
4. **Memory issues**: Reduce population size or number of generations

### Dependencies

```bash
pip install numpy alns
```

### Data Requirements

The framework expects PCTSP instances in pickle format:
- `data/pctsp_20_20_instances.pkl` for 20-node problems
- `data/pctsp_50_20_instances.pkl` for 50-node problems  
- `data/pctsp_100_20_instances.pkl` for 100-node problems

## 🧬 Evolution Strategies

### i1 (Initialization)
Generates initial repair operators from scratch using problem-specific prompts.

### e1 (Exploration)
Creates completely different algorithms inspired by top 2 parents but with novel approaches.

### e2 (Exploitation)
Combines and improves the best ideas from top 2 parent operators.

### m1 (Modification)
Makes significant modifications to the best operator while keeping core structure.

### m3 (Simplification)
Simplifies the best operator to improve generalization and robustness.

## 📝 Example Output

```
🧬 Starting EoH-PCTSP Evolution
Population size: 4
Generations: 3
Problem size: 20
Instances: 2

=== Generating Initial Population (size: 4) ===

Generating individual 1/4
Created operator - Objective: 42.15, Gap: -8.23%, Feasible: True

...

✅ EoH-PCTSP Evolution completed!
Runtime: 12.3 minutes
Best operator objective: 38.92
Best operator gap: -15.67%
Results saved to: eoh_pctsp_results
```

## 🤝 Integration with Main Framework

This PCTSP adaptation reuses the core EoH infrastructure:
- `interface_llm.py`: LLM communication interface
- `llm_api.py`: LLM API wrapper
- Evolution strategies pattern from PFSP version

Key adaptations:
- PCTSP-specific problem encoding and solution representation
- Prize-penalty trade-off considerations in prompts
- PCTSP-specific evaluation metrics and constraints
- Adapted destroy/repair operators for PCTSP structure

## 📚 References

- **EoH Framework**: Evolution of Heuristics methodology
- **PCTSP**: Price Collecting Travelling Salesman Problem
- **ALNS**: Adaptive Large Neighborhood Search
- **LLM**: Large Language Models for algorithm generation 