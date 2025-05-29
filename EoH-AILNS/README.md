# EoH-PFSP: Evolution of Heuristics for Permutation Flow Shop Problem

A simplified Evolution of Heuristics framework for automatically generating and evolving PFSP repair operators using Large Language Models.

## 🎯 Project Overview

This project implements a **simplified and production-ready** EoH framework that:
- ✅ Generates PFSP repair operators using LLMs
- ✅ Evolves operators through multiple strategies (i1, e1, e2, m1, m2, m3)
- ✅ Evaluates performance with proper metrics (objective, gap to BKV, runtime)
- ✅ Stores comprehensive results in JSON format
- ✅ Provides complete evaluation against Taillard benchmark instances

## 🚀 Quick Start

### 1. Generate EoH Operators
```bash
python demo_simple_eoh.py
```

### 2. Evaluate Best Operator
```bash
python quick_evaluation.py
```

### 3. Visualize Results
```bash
python visualize_results.py
```

## 📁 Core Files

### **Framework Implementation**
- `eoh_pfsp.py` - Main EoH framework (simplified)
- `eoh_evolution_pfsp.py` - Evolution engine with 6 strategies
- `pfsp_prompts.py` - Clean, direct prompts for LLM
- `interface_llm.py` - LLM wrapper interface
- `PFSP.py` - Core PFSP functions and classes

### **Evaluation & Testing**
- `quick_evaluation.py` - Fast evaluation on representative instances
- `evaluate_eoh_operators.py` - Comprehensive evaluation on all instances
- `test_simple_eoh.py` - Framework test suite
- `demo_simple_eoh.py` - Usage demonstration

### **Analysis & Results**
- `visualize_results.py` - Results visualization and analysis
- `EoH_Evaluation_Summary.md` - Comprehensive evaluation results
- `quick_eoh_evaluation.csv` - Detailed instance results
- `eoh_evaluation_results.png` - Performance charts

### **Dependencies**
- `llm_api.py` - Core LLM interface
- `data/` - Taillard benchmark instances

## 🔬 Evaluation Results

### **Performance Summary**
| Metric | EoH Operator | Baseline | Improvement |
|--------|--------------|----------|-------------|
| Average Gap to BKV | 0.70% | 0.77% | +0.07 pp |
| Win Rate | 33.3% (3/9) | 66.7% (6/9) | - |
| Best Performance | 0.00% | 0.00% | Tie |
| Worst Performance | 2.35% | 3.34% | EoH better |

### **Key Findings**
- ✅ **Competitive Performance**: EoH operator matches baseline with only 0.07 pp average difference
- ✅ **Better on Larger Instances**: Shows improvement on 50+ job problems (+0.40 pp on 50-job, +0.26 pp on 100-job)
- ✅ **Significant Improvements**: Up to 0.99 pp improvement on complex instances
- ⚠️ **Training Limitation**: Single-instance training limits generalization to smaller problems

### **Performance by Problem Size**
| Problem Size | EoH Gap | Baseline Gap | Improvement | Win Rate |
|--------------|---------|--------------|-------------|----------|
| 20 jobs | 0.45% | 0.18% | -0.27 pp | 0% (0/4) |
| 50 jobs | 1.30% | 1.70% | +0.40 pp | 67% (2/3) |
| 100 jobs | 0.31% | 0.57% | +0.26 pp | 50% (1/2) |

## 🎨 Design Philosophy

> **"Keep it simple, keep it working, like the original PFSP code."**

### **Key Simplifications**
1. **Clear Prompts**: Direct, unambiguous prompts that work
2. **Simple Execution**: `exec(code, globals(), namespace)` like original
3. **No Complex Setup**: Generated operators work in existing context
4. **Clean Architecture**: Following original PFSP patterns

### **Technical Approach**
- **Global Data Access**: Operators access `DATA`, `Solution`, etc. automatically
- **Robust Error Handling**: Fallback operators for failed generations
- **JSON Storage**: All results with proper type conversion
- **Production Ready**: No complex dependencies or setup

## 🧬 Evolution Strategies

| Strategy | Description | Usage |
|----------|-------------|--------|
| **i1** | Generate initial operators from scratch | Population initialization |
| **e1** | Create totally different algorithms | Exploration |
| **e2** | Combine and improve existing operators | Exploitation |
| **m1** | Modify existing algorithms significantly | Refinement |
| **m2** | Tune parameters and constants | Fine-tuning |
| **m3** | Simplify for better generalization | Robustness |

## 📊 Comprehensive Testing

### **Framework Tests** (✅ All Pass)
```bash
python test_simple_eoh.py
```
- ✅ Initial population generation
- ✅ Evolution strategies 
- ✅ JSON storage and loading
- ✅ Operator creation and evaluation

### **Performance Evaluation**
```bash
python quick_evaluation.py      # 9 representative instances
python evaluate_eoh_operators.py # 80 full Taillard instances
```

## 📈 Results Structure

```
demo_results/
├── generations/
│   ├── generation_0.json    # Initial population
│   ├── generation_1.json    # First evolution
│   └── ...
├── best/
│   ├── best_generation_0.json
│   └── ...
└── final_summary.json       # Complete run summary
```

Each operator stored with:
- Algorithm description
- Complete code
- Performance metrics (objective, gap, runtime)
- Feasibility status
- Generation and strategy info
- Timestamp

## 🔧 Usage Examples

### **Basic Usage**
```python
from eoh_pfsp import EoH_PFSP

# Initialize framework
eoh = EoH_PFSP(
    pop_size=4,
    n_generations=3,
    data_file="data/j50_m20/j50_m20_01.txt"
)

# Run complete evolution
final_population = eoh.run()

# Get best operator
best = final_population[0]
print(f"Best gap: {best['gap']:.2f}%")
```

### **Custom Configuration**
```python
eoh = EoH_PFSP(
    pop_size=6,           # Larger population
    n_generations=5,      # More evolution
    data_file="data/j100_m10/j100_m10_01.txt",  # Larger problem
    output_dir="my_results"
)
```

## 🎯 Framework Achievements

### **✅ All Requirements Met**
1. **Performance Metrics Logging**: ✅ Objective values, gaps to BKV, runtimes
2. **LLM Evolution of Top Operators**: ✅ 6 evolution strategies, parent selection
3. **Comprehensive Storage**: ✅ JSON with all generations, proper type conversion
4. **Complete EoH Process**: ✅ Initialization → Evolution → Evaluation → Storage

### **✅ Technical Excellence**
- **No Namespace Issues**: Uses globals like original code
- **No Data Problems**: Seamless access to PFSP functions
- **Clean Error Handling**: Robust fallbacks for LLM failures
- **Production Ready**: Simple, maintainable, extensible

### **✅ Scientific Rigor**
- **Proper Baselines**: Fair comparison with equivalent operators
- **Statistical Analysis**: Win rates, improvement distributions
- **Comprehensive Testing**: Multiple instance sizes and types
- **Reproducible Results**: Documented methodology and parameters

## 📚 Documentation

- **README_Simple_EoH.md**: Framework technical details
- **EoH_Evaluation_Summary.md**: Complete evaluation results
- **Code Comments**: Extensive inline documentation
- **Test Suite**: Comprehensive validation

## 🚀 Future Enhancements

1. **Multi-Instance Training**: Train on diverse problem sizes
2. **Advanced Strategies**: More sophisticated evolution approaches
3. **Ensemble Methods**: Combine multiple EoH operators
4. **Adaptive Parameters**: Dynamic population and generation sizing
5. **Real-time Evaluation**: Live performance tracking during evolution

## 🏆 Conclusion

The EoH-PFSP framework successfully demonstrates:
- **Feasibility**: LLMs can generate competitive PFSP operators
- **Simplicity**: Clean implementation following proven patterns
- **Effectiveness**: Competitive performance, especially on larger instances
- **Extensibility**: Easy to modify and enhance

**Result**: A production-ready Evolution of Heuristics framework that generates and evaluates PFSP repair operators, proving the concept while maintaining simplicity and effectiveness.

---

*EoH-PFSP Framework - Where Simplicity Meets Sophistication* 🧬✨ 