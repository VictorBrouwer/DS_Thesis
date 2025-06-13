# Multi-Size PCTSP Operator Evolution and Evaluation

This directory contains scripts for evolving PCTSP repair operators on different problem sizes and evaluating them comprehensively.

## Overview

The multi-size experiment consists of:

1. **Evolution Phase**: Evolve repair operators on 3 different problem sizes (20, 50, 100 nodes)
2. **Evaluation Phase**: Test all 3 evolved operators together on all problem sizes
3. **Analysis Phase**: Compare performance across different sizes and operators

## Files Created

### Evolution Scripts
- `run_evolution_size_50.py` - Run evolution for 50-node problems
- `run_evolution_size_100.py` - Run evolution for 100-node problems  
- `run_complete_multi_size_experiment.py` - Master script to run everything

### Evaluation Scripts
- `evaluate_multi_size_operators.py` - Evaluate all 3 operators together
- `test_multi_size_setup.py` - Test script to verify setup

### Documentation
- `README_multi_size.md` - This file

## Prerequisites

1. **Size 20 Evolution Complete**: You must have already run evolution for 20-node problems:
   ```bash
   python run_eoh_pctsp.py --problem_size 20
   ```
   This should create `eoh_pctsp_results/final_summary.json`

2. **Required Dependencies**: PCTSP module, ALNS library, pandas, numpy

## Quick Start

### Option 1: Run Everything Automatically
```bash
python run_complete_multi_size_experiment.py
```

This will:
- Check existing results
- Run evolution for size 50 (if needed)
- Run evolution for size 100 (if needed)  
- Evaluate all 3 operators together

### Option 2: Run Step by Step

1. **Test Setup**:
   ```bash
   python test_multi_size_setup.py
   ```

2. **Run Evolution for Size 50**:
   ```bash
   python run_evolution_size_50.py
   ```

3. **Run Evolution for Size 100**:
   ```bash
   python run_evolution_size_100.py
   ```

4. **Evaluate All Operators**:
   ```bash
   python evaluate_multi_size_operators.py
   ```

## Output Structure

### Evolution Results
```
eoh_pctsp_results/          # Size 20 (existing)
├── final_summary.json
├── best/
└── generations/

eoh_pctsp_results_50/       # Size 50 (new)
├── final_summary.json
├── best/
└── generations/

eoh_pctsp_results_100/      # Size 100 (new)
├── final_summary.json
├── best/
└── generations/
```

### Evaluation Results
```
multi_size_results/
├── multi_size_pctsp_evaluation_YYYYMMDD_HHMMSS.csv
└── multi_size_pctsp_summary_YYYYMMDD_HHMMSS.json
```

## Results Analysis

### CSV File Contents
The evaluation CSV contains detailed results for each instance:
- `test`: Instance identifier
- `size`: Problem size (20, 50, 100)
- `total_prize_required`: Required prize for the instance
- `initial_objective`: Initial solution objective
- `final_objective`: Best objective found by multi-operator ALNS
- `improvement`: Objective improvement
- `feasible`: Whether final solution is feasible
- `tour_length`: Length of final tour
- `prize_collected`: Total prize collected
- `runtime`: Time taken (seconds)

### JSON Summary Contents
The summary JSON provides aggregated statistics:
- Overall performance metrics
- Performance by problem size
- Operator information
- Runtime statistics

### Key Metrics
- **Improvement Rate**: Percentage of instances where operators found better solutions
- **Average Improvement**: Mean objective improvement across all instances
- **Feasible Rate**: Percentage of feasible solutions found
- **Runtime**: Average time per instance

## Expected Timeline

- **Size 50 Evolution**: ~10 minutes (5 pop × 4 gen × 0.5 min)
- **Size 100 Evolution**: ~10 minutes (5 pop × 4 gen × 0.5 min) 
- **Multi-Size Evaluation**: ~60 minutes (60 instances × 1 min each)
- **Total**: ~80 minutes

## Troubleshooting

### Common Issues

1. **"Size 20 results not found"**
   ```bash
   python run_eoh_pctsp.py --problem_size 20
   ```

2. **Import errors**
   - Check PCTSP module is in the correct location
   - Verify ALNS library is installed
   - Ensure all dependencies are available

3. **Data loading errors**
   - Verify `data/` directory contains the pickle files
   - Check file permissions

4. **Evolution failures**
   - Check LLM API credentials if using external LLM
   - Verify sufficient disk space for results
   - Check network connectivity for LLM calls

### Test Script
Always run the test script first to verify setup:
```bash
python test_multi_size_setup.py
```

## Comparison with PFSP

This implementation follows the same pattern as `EoH-AILNS/PFSP/evaluate_multi_size_operators.py` but adapted for PCTSP:

### Similarities
- Same overall structure and workflow
- Similar ALNS configuration
- Comparable evaluation metrics
- Same result saving format

### Differences
- PCTSP-specific solution creation and evaluation
- Different feasibility checking
- Prize collection metrics
- Problem-specific operators and data loading

## Next Steps

After running the multi-size evaluation:

1. **Analyze Results**: Compare the performance of operators evolved on different sizes
2. **Identify Patterns**: Look for size-specific specialization vs. generalization
3. **Further Experiments**: Consider additional problem sizes or different evolution parameters
4. **Publication**: Use results for research papers or reports

## Technical Details

### Operator Loading
Each evolved operator is loaded from its respective `final_summary.json` file and executed in a safe namespace to create the repair function.

### ALNS Configuration
- **Destroy Operators**: Random removal, worst removal
- **Repair Operators**: 3 evolved operators (small, medium, large)
- **Selection**: AlphaUCB with auto-tuning
- **Acceptance**: Simulated Annealing with auto-fitting
- **Termination**: 60 seconds runtime per instance

### Performance Metrics
- **Objective Improvement**: Difference between initial and final objectives
- **Feasibility**: Whether solutions satisfy PCTSP constraints
- **Runtime**: Time efficiency of the multi-operator approach
- **Generalization**: How well operators perform across different problem sizes 