# 🧬 scAgent: AI-Powered Single-Cell Analysis for sc-eQTL

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-316192.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**scAgent** is a high-performance bioinformatics tool for identifying and filtering human single-cell RNA-seq datasets suitable for single-cell expression quantitative trait loci (sc-eQTL) analysis. It leverages AI to intelligently process millions of genomic datasets with unprecedented speed and accuracy.

## ✨ Key Features

- 🚀 **10x Faster Processing** with parallel AI execution (v2.0)
- 🤖 **AI-Powered Detection** for ambiguous RNA-seq classification
- 🧬 **Robust Human Sample Identification** from multiple metadata fields
- 🚫 **Automatic Cell Line Exclusion** with pattern recognition
- 📊 **10 Key sc-eQTL Criteria Extraction** for comprehensive analysis
- 📈 **Real-time Progress Visualization** with performance metrics
- 💾 **Comprehensive CSV Output** with all filtering results

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/scAgent.git
cd scAgent

# Install dependencies
pip install -r requirements.txt

# Configure database connection
cp scAgent/settings_template.yml scAgent/settings.yml
# Edit settings.yml with your database credentials
```

### Basic Usage

```bash
# Test run with 1000 samples
python comprehensive_human_sc_analysis.py --test-run

# Quick scan of 10,000 samples
python comprehensive_human_sc_analysis.py --quick-scan 10000

# Full database scan with maximum performance
python comprehensive_human_sc_analysis.py --fullscan --ai-workers 15
```

## 📊 Performance

With v2.0's parallel processing:

- **1,000 samples**: ~15 seconds (was 2.5 minutes)
- **10,000 samples**: ~2.5 minutes (was 25 minutes)
- **100,000 samples**: ~25 minutes (was 4+ hours)

## 🔧 Advanced Usage

### Custom Configuration

```bash
# Process specific number of samples with custom output
python comprehensive_human_sc_analysis.py \
    --max-samples 50000 \
    --batch-size 5000 \
    --output results_50k.csv \
    --ai-workers 10
```

### Filtering Pipeline

1. **Human Detection**: Multi-field validation including mixed species handling
2. **Cell Line Exclusion**: Pattern-based detection from characteristics
3. **Single-Cell Identification**: Direct markers + AI for boundary cases
4. **sc-eQTL Criteria Extraction**: 10 key fields for downstream analysis

## 📋 Output Format

The tool generates two files:

- `comprehensive_human_sc_analysis_YYYYMMDD_HHMMSS.csv`: Complete results with all 82 columns + filtering metadata
- `comprehensive_human_sc_analysis_YYYYMMDD_HHMMSS_statistics.json`: Processing statistics and performance metrics

### Key Output Fields

- All original database columns
- Filter pass/fail status with confidence scores
- 10 extracted sc-eQTL criteria
- AI assistance indicators
- Processing time per sample

## 🏗️ Architecture

```
scAgent/
├── comprehensive_human_sc_analysis.py    # Main analysis script
├── scAgent/
│   ├── utils_sra_geo_ft2_optimizer.py  # Core filtering logic
│   ├── db/                              # Database handlers
│   └── models/                          # AI model integration
└── find_human_samples_test.py           # Testing utilities
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use scAgent in your research, please cite:

```
scAgent: AI-Powered Single-Cell Analysis for sc-eQTL
https://github.com/YimingShi2004/scAgent_v2.0
```

##  Acknowledgments

- PostgreSQL for robust data storage
- Qwen team for LLM capabilities
- Single-cell genomics community
- Yiming SHI & Yanglab & Wuhan Zhang

For detailed documentation, see [README_COMPREHENSIVE_ANALYSIS.md](README_COMPREHENSIVE_ANALYSIS.md) 
