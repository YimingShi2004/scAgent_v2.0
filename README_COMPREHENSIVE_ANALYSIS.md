# 🧬 scAgent: Comprehensive Human Single-Cell Analysis for sc-eQTL

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-316192.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

**scAgent** is an advanced bioinformatics tool designed to identify and filter human single-cell RNA-seq datasets suitable for single-cell expression quantitative trait loci (sc-eQTL) analysis. It processes the `sra_geo_ft2` table containing millions of genomic datasets with intelligent filtering, AI-assisted classification, and comprehensive quality assessment.

### 🎯 Key Features

- **Intelligent Human Sample Detection**: Robust identification of human samples from multiple metadata fields
- **Cell Line Exclusion**: Automatic detection and removal of cell line data
- **Single-Cell Technology Recognition**: Identifies scRNA-seq and scATAC-seq experiments, including ambiguous cases
- **AI-Powered Analysis**: Uses LLM for boundary cases where "RNA-Seq" might actually be single-cell
- **10 Key sc-eQTL Criteria Extraction**: Automated extraction of essential metadata for eQTL analysis
- **High-Performance Processing**: Optimized batch processing with configurable parameters
- **Beautiful Structured Output**: Clean, formatted progress reports and comprehensive CSV results

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# PostgreSQL with sra_geo_ft2 table
psql --version

# Required Python packages
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scAgent.git
cd scAgent

# Install package
pip install -e .
```

### Basic Usage

```bash
# Test run with 1000 samples
python comprehensive_human_sc_analysis.py --test-run

# Quick scan of 10,000 samples
python comprehensive_human_sc_analysis.py --quick-scan 10000

# Full database scan
python comprehensive_human_sc_analysis.py --fullscan

# Custom configuration
python comprehensive_human_sc_analysis.py \
    --max-samples 50000 \
    --batch-size 5000 \
    --output my_results.csv
```

## 📊 Output Format

### CSV Output Columns

The analysis produces a comprehensive CSV file with the following information:

#### Sample Identifiers
- `sra_ID`: Primary SRA identifier
- `gsm_title`: GEO Sample title
- `gse_title`: GEO Series title
- `run_alias`, `experiment_alias`: Additional identifiers

#### Organism Information
- `organism_ch1`: Primary organism field
- `scientific_name`: Scientific species name
- `organism`: Alternative organism field

#### Experiment Details
- `experiment_title`: Full experiment title
- `study_title`: Study title
- `study_abstract`: Study abstract
- `summary`: Additional summary information

#### Technical Information
- `library_strategy`: Sequencing strategy
- `library_source`: Library source
- `platform`: Sequencing platform
- `instrument_model`: Specific instrument used
- `technology`: Technology type

#### Filtering Results
- `passes_all_filters`: Boolean indicating if sample passed all filters
- `final_confidence_score`: Overall confidence score (0-1)
- `rejection_reason`: Reason for rejection if applicable
- `[filter]_passed`: Individual filter results
- `[filter]_confidence`: Confidence for each filter

#### sc-eQTL Criteria (10 Key Fields)
1. `eqtl_organism`: Confirmed organism
2. `eqtl_tissue_type`: Tissue or organ source
3. `eqtl_cell_type`: Specific cell type
4. `eqtl_sample_size`: Number of samples/cells
5. `eqtl_sequencing_platform`: Platform used
6. `eqtl_project_id`: Project identifiers
7. `eqtl_publication_info`: PMID if published
8. `eqtl_geographic_location`: Sample origin
9. `eqtl_age_range`: Age information
10. `eqtl_disease_status`: Disease or healthy status

### Statistics JSON

A companion `_statistics.json` file provides:
- Processing statistics
- Performance metrics
- Error logs
- Summary analysis

## 🔧 Advanced Configuration

### Command Line Options

```bash
usage: comprehensive_human_sc_analysis.py [-h] [--max-samples MAX_SAMPLES]
                                         [--batch-size BATCH_SIZE]
                                         [--output OUTPUT] [--no-ai]
                                         [--test-run] [--fullscan]
                                         [--quick-scan QUICK_SCAN]

Comprehensive Human Single-Cell Analysis for sc-eQTL

optional arguments:
  -h, --help            show this help message and exit
  --max-samples MAX_SAMPLES
                        Maximum samples to process (default: all)
  --batch-size BATCH_SIZE
                        Batch size for processing (default: 5000)
  --output OUTPUT       Output CSV file (default: auto-generated with timestamp)
  --no-ai               Disable AI-assisted analysis
  --test-run            Run with limited samples for testing (1000 samples)
  --fullscan            Perform full database scan (all records)
  --quick-scan QUICK_SCAN
                        Quick scan with specified number of samples (e.g., 10000)
```

### Filtering Logic

1. **Human Detection Priority**:
   - `organism_ch1` (most reliable)
   - `scientific_name`
   - `experiment_title` (parsed for "Homo sapiens")
   - Text fields (study_title, abstract, etc.)

2. **Cell Line Exclusion**:
   - Checks `characteristics_ch1` for "cell line:" patterns
   - Examines `source_name_ch1` for common cell line names
   - Scans description fields for cell line indicators

3. **Single-Cell Detection**:
   - Direct keyword matching (scRNA-seq, scATAC-seq, 10X, etc.)
   - Platform-specific identifiers
   - AI-assisted analysis for ambiguous "RNA-Seq" cases

## 📈 Performance Optimization

### Batch Processing
- Default batch size: 5,000 records
- Full scan batch size: 10,000 records
- Memory-efficient streaming processing

### Parallel Processing
- Concurrent database queries
- Batch-level parallelization
- AI inference optimization

### Expected Performance
- ~100-500 samples/second (depending on AI usage)
- 10,000 samples: ~1-2 minutes
- 100,000 samples: ~10-20 minutes
- Full database scan: 2-4 hours

## 🤖 AI Integration

The system uses Qwen-max LLM for:
- Ambiguous RNA-seq classification
- Geographic location extraction
- Age range identification
- Disease status determination

### AI Configuration

```python
# In scAgent/settings.yml
model_name: "Qwen3-235B-A22B"
model_api_base: "http://your-api-endpoint/v1"
```

## 📝 Example Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║        COMPREHENSIVE HUMAN SINGLE-CELL ANALYSIS FOR sc-eQTL v1.0             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Configuration:                                                               ║
║   • Start Time         : 2024-01-15 10:30:45                                ║
║   • Max Samples        :      10000                                          ║
║   • Batch Size         :       2000                                          ║
║   • AI Assistance      :    ENABLED                                          ║
║   • Output File        : human_sc_analysis_20240115_103045.csv              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                          PROGRESS STATISTICS                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Processed        :      10,000 samples                                 ║
║ Human Samples          :       8,542 confirmed                               ║
║ Single-Cell Identified :       1,234 experiments                             ║
║ Cell Lines Excluded    :         456 samples                                 ║
║ Passed All Filters     :         987 datasets                                ║
║ AI-Assisted Decisions  :         234 cases                                   ║
║ Processing Errors      :           5 errors                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Overall Pass Rate      :       9.87% of all samples                          ║
║ Single-Cell Rate       :      14.44% of human samples                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check PostgreSQL connection
   psql -h localhost -U your_user -d your_database
   ```

2. **Column Mapping Issues**
   ```bash
   # Verify table structure
   python inspect_sra_geo_ft2_table.py
   ```

3. **AI Service Unavailable**
   ```bash
   # Run without AI
   python comprehensive_human_sc_analysis.py --no-ai
   ```

## 📚 Technical Details

### Database Schema
- Table: `merged.sra_geo_ft2`
- Columns: 82 (mapped dynamically)
- Primary key: `sra_ID`

### Column Mapping Strategy
- Logical-to-physical mapping
- Fuzzy matching for flexibility
- Automatic column discovery

### Quality Scoring
- Human detection: 0.95 confidence
- Cell line exclusion: 0.90 confidence
- Single-cell detection: 0.60-0.95 confidence
- Overall threshold: 0.70

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PostgreSQL for robust data storage
- Qwen team for LLM capabilities
- Single-cell genomics community

---

**For questions or support, please open an issue on GitHub.** 