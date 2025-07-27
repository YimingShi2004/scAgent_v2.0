# 🧬 scAgent Scripts Inventory for GitHub

## 📁 Project Structure and Files

### 1. Core Module Files (`scAgent/`)

#### Database Connection & Handling
- `scAgent/db/__init__.py` - Database module initialization
- `scAgent/db/connect.py` - PostgreSQL connection management
- `scAgent/db/merged_table_handler.py` - Merged table operations handler
- `scAgent/db/query.py` - Query builder utilities
- `scAgent/db/schema.py` - Database schema definitions

#### Models & AI Integration
- `scAgent/models/__init__.py` - Models module initialization
- `scAgent/models/base.py` - Base model interface
- `scAgent/models/client.py` - Model client interface
- `scAgent/models/qwen_client.py` - Qwen LLM client implementation

#### Utilities & Filtering
- `scAgent/utils.py` - Basic utilities
- `scAgent/utils_enhanced_filtering.py` - Enhanced filtering system
- `scAgent/utils_sra_geo_ft2_optimizer.py` ⭐ - **Main optimizer for sra_geo_ft2 table**
- `scAgent/utils_species_enhanced.py` - Species detection utilities
- `scAgent/utils_improved_assessment.py` - Assessment utilities

#### CLI Interface
- `scAgent/cli/main.py` - CLI entry point
- `scAgent/cli/commands.py` - CLI commands
- `scAgent/cli/enhanced_commands.py` - Enhanced CLI commands

#### Configuration
- `scAgent/settings.yml` - Configuration settings
- `scAgent/__init__.py` - Package initialization
- `scAgent/__main__.py` - Package main entry

### 2. Main Execution Scripts

#### Primary Analysis Scripts
- `comprehensive_human_sc_analysis.py` ⭐ - **Main comprehensive analysis script (v2.0 with parallel processing)**
- `run_sra_geo_ft2_analysis.py` - Production analysis runner
- `find_human_samples_test.py` - Human sample testing and validation

#### Testing Scripts
- `test_sra_geo_ft2_optimizer.py` - Optimizer unit tests
- `test_enhanced_human_filtering.py` - Enhanced filtering tests
- `test_fixed_human_detection.py` - Human detection tests
- `test_full_comprehensive_clean.py` - Full integration tests

#### Utility Scripts
- `inspect_sra_geo_ft2_table.py` - Table structure inspection
- `list_tables.py` - Database table listing
- `quick_db_test.py` - Quick database connection test

### 3. Documentation Files

- `README_COMPREHENSIVE_ANALYSIS.md` ⭐ - **Main documentation**
- `README.md` - Basic readme
- `QUICKSTART.md` - Quick start guide
- `USER_GUIDE.md` - User guide
- `SCRIPTS_INVENTORY.md` - This file

### 4. Configuration Files

- `pyproject.toml` - Python project configuration
- `.gitignore` - Git ignore patterns (create if not exists)
- `requirements.txt` - Python dependencies (generate from environment)

### 5. Sample Output Files (Optional - for examples)

- `comprehensive_human_sc_analysis_YYYYMMDD_HHMMSS.csv` - Sample output
- `comprehensive_human_sc_analysis_YYYYMMDD_HHMMSS_statistics.json` - Sample statistics

## 📋 Key Files Summary

### Essential Files for GitHub Upload:

```
scAgent/
├── scAgent/                              # Core package
│   ├── __init__.py
│   ├── settings.yml
│   ├── db/                              # Database module
│   │   ├── __init__.py
│   │   ├── connect.py
│   │   └── merged_table_handler.py
│   ├── models/                          # AI models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── client.py
│   │   └── qwen_client.py
│   └── utils_sra_geo_ft2_optimizer.py  # Core optimizer
├── comprehensive_human_sc_analysis.py    # Main script
├── find_human_samples_test.py           # Testing tool
├── inspect_sra_geo_ft2_table.py        # Table inspector
├── README_COMPREHENSIVE_ANALYSIS.md      # Documentation
├── pyproject.toml                       # Project config
└── requirements.txt                     # Dependencies
```

## 🚀 Quick Setup Commands

### 1. Generate requirements.txt
```bash
pip freeze > requirements.txt
```

### 2. Create .gitignore
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Data files
*.csv
*.json
*.db
*.xlsx

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
scAgent.egg-info/
dist/
build/
*.egg
EOF
```

### 3. Initialize Git repository
```bash
git init
git add .
git commit -m "Initial commit: scAgent v2.0 with parallel processing"
```

## 📝 Notes for GitHub Upload

1. **Sensitive Information**: 
   - Remove any API keys from `settings.yml`
   - Use environment variables for credentials

2. **Large Files**:
   - Don't upload sample data CSV files
   - Use Git LFS for large test datasets if needed

3. **Documentation**:
   - Rename `README_COMPREHENSIVE_ANALYSIS.md` to `README.md` for GitHub
   - Add LICENSE file (MIT recommended)

4. **Dependencies**:
   - Ensure `requirements.txt` includes:
     - psycopg2-binary
     - pandas
     - tqdm
     - requests
     - dynaconf
     - rich (for colored output)

5. **Testing**:
   - Add GitHub Actions workflow for automated testing
   - Include sample test data (small subset)

## 🏷️ Recommended GitHub Topics

- `single-cell`
- `scrna-seq`
- `eqtl`
- `bioinformatics`
- `genomics`
- `python`
- `postgresql`
- `data-processing`
- `parallel-processing`
- `ai-assisted` 