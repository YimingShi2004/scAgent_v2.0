# Enhanced sc-eQTL Analysis System v2.2 Jul 30/2025

## Overview

The Enhanced sc-eQTL Analysis System is a comprehensive solution for discovering and analyzing human single-cell datasets suitable for sc-eQTL (single-cell expression quantitative trait loci) analysis. This system provides advanced filtering, metadata extraction, and data access capabilities.

## Key Features

### 🔗 SRA Lite URL Generation
- **Automatic URL Generation**: Generates SRA Lite URLs for data access
- **Download Options**: Includes both data access and download URLs
- **Configurable Output**: Can include or exclude download URLs based on user preference

### 🧬 Enhanced Dataset Classification
- **scRNA-seq vs scATAC-seq**: Intelligent classification of single-cell experiment types
- **Technology Detection**: Identifies specific technologies (10x, Smart-seq, Drop-seq, etc.)
- **Confidence Scoring**: Provides confidence scores for classifications

### 🏥 Tumor vs Normal Tissue Detection
- **Automatic Detection**: Identifies tumor samples vs normal tissue
- **Tumor Type Classification**: Categorizes specific tumor types
- **AI-Assisted Analysis**: Uses AI for sophisticated tumor detection

### 🧪 Cell Line Detection and Exclusion
- **Comprehensive Detection**: Identifies cell line samples using multiple indicators
- **Automatic Exclusion**: Excludes cell lines from sc-eQTL analysis
- **Configurable Rules**: Customizable detection patterns

### 📊 Age Information Extraction
- **Multiple Formats**: Handles various age formats (years, months, days, postnatal days)
- **Pattern Recognition**: Uses regex patterns to extract age information
- **Source Tracking**: Tracks the source field of age information

### 📚 PMC Document Analysis
- **Full-Text Extraction**: Extracts detailed information from PMC PDFs
- **AI-Powered Analysis**: Uses AI to analyze publication content
- **Comprehensive Metadata**: Extracts age, sample size, geographic location, disease status

### 🚀 GPU-Accelerated Processing
- **Parallel Processing**: Multi-threaded processing for high performance
- **GPU Support**: Optional GPU acceleration for AI operations
- **Memory Optimization**: Efficient memory management for large datasets

### 📈 Quality Assessment
- **Multi-dimensional Scoring**: Comprehensive quality metrics
- **Configurable Thresholds**: Adjustable quality thresholds
- **Performance Monitoring**: Real-time performance tracking

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database
- GPU support (optional, for acceleration)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd scAgent_v2.1

# Install enhanced requirements
pip install -r requirements_enhanced.txt

# Configure database connection
cp scAgent_2025/settings_enhanced.yml scAgent_2025/settings.yml
# Edit settings.yml with your database credentials
```

## Configuration

### Database Configuration
```yaml
database:
  host: "localhost"
  port: 5432
  database: "scagent_db"
  user: "scagent_user"
  password: "${DB_PASSWORD}"
  schema: "merged"
  table: "sra_geo_ft2"
```

### AI Model Configuration
```yaml
ai_models:
  primary:
    model_name: "Qwen3-235B-A22B"
    api_base: "http://10.28.1.21:30080/v1"
    max_tokens: 32000
    temperature: 0.2
```

### Processing Configuration
```yaml
processing:
  max_ai_workers: 20
  max_db_workers: 10
  batch_size: 5000
  enable_gpu: true
```

## Usage

### Basic Usage

```python
from enhanced_sc_eqtl_analysis import EnhancedScEqtlAnalysis

# Initialize analysis system
analysis = EnhancedScEqtlAnalysis(
    max_ai_workers=20,
    enable_gpu=True,
    include_download_urls=True
)

# Run comprehensive analysis
result = analysis.run_enhanced_analysis(
    max_samples=10000,
    batch_size=5000,
    output_file="enhanced_results.csv",
    enable_ai=True
)
```

### Command Line Usage

```bash
# Quick test run
python enhanced_sc_eqtl_analysis.py --test-run

# Process specific number of samples
python enhanced_sc_eqtl_analysis.py --quick-scan 10000 --batch-size 2000

# Full database scan with download URLs
python enhanced_sc_eqtl_analysis.py --fullscan --include-downloads

# High-performance mode
python enhanced_sc_eqtl_analysis.py --quick-scan 50000 --ai-workers 20
```

### Advanced Usage

```python
# Custom configuration
analysis = EnhancedScEqtlAnalysis(
    max_ai_workers=15,
    enable_gpu=False,
    include_download_urls=False
)

# Process with custom filters
result = analysis.run_enhanced_analysis(
    max_samples=5000,
    batch_size=1000,
    output_file="custom_results.csv",
    enable_ai=True
)
```

## Output Format

### CSV Output Fields

The enhanced analysis produces a comprehensive CSV file with the following fields:

#### Basic Information
- `sra_ID`: SRA run identifier
- `gsm_title`: GEO sample title
- `gse_title`: GEO series title
- `experiment_title`: Experiment title
- `organism_ch1`: Organism information

#### Enhanced Metadata
- `experiment_type`: scRNA-seq or scATAC-seq
- `sc_technology`: Specific technology used
- `is_tumor`: Tumor status (True/False)
- `tumor_type`: Specific tumor type
- `is_cell_line`: Cell line status (True/False)
- `cell_line_name`: Cell line name if applicable

#### Age Information
- `age_value`: Extracted age value
- `age_unit`: Age unit (years, months, days, etc.)
- `age_source`: Source field of age information
- `age_confidence`: Confidence score for age extraction

#### Sample Information
- `estimated_sample_size`: Estimated sample size
- `sample_size_source`: Source of sample size information
- `sample_size_confidence`: Confidence score for sample size

#### Publication Information
- `pmid`: PubMed ID
- `doi`: Digital Object Identifier
- `journal`: Journal name
- `publication_date`: Publication date
- `authors`: Author information

#### Geographic and Demographic
- `geographic_location`: Geographic location
- `ethnicity`: Ethnicity information
- `gender`: Gender information
- `health_status`: Health status

#### Quality Metrics
- `data_completeness`: Data completeness score
- `metadata_richness`: Metadata richness score
- `overall_quality_score`: Overall quality score
- `final_confidence_score`: Final confidence score

#### URLs
- `sra_lite_url`: SRA Lite data access URL
- `data_access_url`: Data access URL
- `fastq_download_url`: FASTQ download URL (if enabled)
- `sra_download_url`: SRA download URL (if enabled)

#### Filter Results
- `passes_all_filters`: Overall filter result
- `human_check_passed`: Human sample check
- `cell_line_check_passed`: Cell line exclusion check
- `single_cell_check_passed`: Single-cell experiment check
- `tumor_check_passed`: Tumor status check
- `quality_check_passed`: Quality threshold check

### Statistics Output

The system also generates a JSON statistics file with:
- Processing statistics
- Quality metrics
- Performance data
- Error summaries

## PMC Analysis

### Features
- **Automatic PMC ID Discovery**: Finds PMC IDs from PMIDs
- **Full-Text Extraction**: Extracts complete text from PMC documents
- **AI-Powered Analysis**: Uses AI to analyze publication content
- **Comprehensive Metadata**: Extracts detailed information

### Usage
```python
from scAgent_2025.utils_pmc_analyzer import PmcAnalyzer

# Initialize PMC analyzer
pmc_analyzer = PmcAnalyzer()

# Analyze single PMID
result = pmc_analyzer.analyze_pmid_comprehensive("12345678")

# Batch analyze multiple PMIDs
pmids = ["12345678", "87654321", "11223344"]
results = pmc_analyzer.batch_analyze_pmids(pmids, max_workers=5)
```

## Performance Optimization

### Parallel Processing
- Multi-threaded database queries
- Parallel AI analysis
- Batch processing optimization

### Memory Management
- Efficient memory usage
- Garbage collection optimization
- Memory monitoring

## Error Handling

### Retry Logic
- Automatic retry for failed operations
- Exponential backoff
- Configurable retry limits

### Error Reporting
- Detailed error logging
- Error categorization
- Performance impact tracking

## Monitoring and Logging

### Logging Configuration
- Configurable log levels
- File rotation
- Performance metrics logging

### Progress Tracking
- Real-time progress updates
- Batch processing statistics
- Performance monitoring

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check database credentials
   - Verify network connectivity
   - Ensure database is running

2. **AI Model Errors**
   - Verify API endpoints
   - Check API keys
   - Monitor rate limits

3. **Memory Issues**
   - Reduce batch size
   - Enable garbage collection
   - Monitor memory usage

4. **Performance Issues**
   - Increase worker count
   - Enable GPU acceleration
   - Optimize database queries

### Debug Mode
```bash
# Enable debug mode
python enhanced_sc_eqtl_analysis.py --debug

# Verbose output
python enhanced_sc_eqtl_analysis.py --verbose
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_enhanced.txt

# Run tests
pytest tests/

# Code formatting
black scAgent_2025/
flake8 scAgent_2025/
```

### Adding New Features
1. Create feature branch
2. Implement functionality
3. Add tests
4. Update documentation
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review troubleshooting guide

## Changelog

### Version 2.2
- Added SRA Lite URL generation
- Enhanced age information extraction
- Added tumor vs normal tissue detection
- Improved scRNA-seq vs scATAC-seq classification
- Added cell line detection and exclusion
- Integrated PMC document analysis
- Added GPU acceleration support
- Enhanced quality assessment
- Improved parallel processing
- Added comprehensive error handling 
