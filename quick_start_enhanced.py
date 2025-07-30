#!/usr/bin/env python3
"""
Quick Start Script for Enhanced sc-eQTL Analysis
Demonstrates how to use the enhanced analysis system with all new features.
"""

import sys
import os
import time
from datetime import datetime

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent_2025'))

def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ENHANCED sc-eQTL ANALYSIS QUICK START                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🚀 Features:                                                               ║
║  • SRA Lite URL generation with download options                           ║
║  • Age information extraction from GEO/SRA characteristics                 ║
║  • Tumor vs normal tissue discrimination                                   ║
║  • scRNA-seq vs scATAC-seq classification                                 ║
║  • Cell line detection and exclusion                                       ║
║  • PMC document analysis for detailed information                          ║
║  • GPU-accelerated processing                                              ║
║  • Comprehensive metadata extraction                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def demo_basic_usage():
    """Demonstrate basic usage of the enhanced system."""
    print("🔧 Basic Usage Demo")
    print("-" * 50)
    
    try:
        from enhanced_sc_eqtl_analysis import EnhancedScEqtlAnalysis
        
        # Initialize the enhanced analysis system
        print("Initializing Enhanced Analysis System...")
        analysis = EnhancedScEqtlAnalysis(
            max_ai_workers=10,  # Reduced for demo
            enable_gpu=True,
            include_download_urls=True
        )
        
        print("✅ System initialized successfully!")
        
        # Run a small test analysis
        print("\nRunning test analysis (100 samples)...")
        result = analysis.run_enhanced_analysis(
            max_samples=100,
            batch_size=50,
            output_file="demo_results.csv",
            enable_ai=True
        )
        
        if result["status"] == "completed":
            print("✅ Test analysis completed successfully!")
            print(f"📊 Results: {result['results']['passed_all_filters']} datasets found")
            print(f"📈 Pass rate: {result['results']['overall_pass_rate']:.1f}%")
        else:
            print("❌ Test analysis failed")
            
    except Exception as e:
        print(f"❌ Error in basic usage demo: {e}")

def demo_individual_features():
    """Demonstrate individual enhanced features."""
    print("\n🔬 Individual Features Demo")
    print("-" * 50)
    
    try:
        from scAgent_2025.utils_enhanced_sc_eqtl_optimizer import EnhancedScEqtlOptimizer
        
        optimizer = EnhancedScEqtlOptimizer()
        
        # Demo 1: SRA Lite URL Generation
        print("\n1. SRA Lite URL Generation:")
        sra_id = "ERR4405370"
        urls = optimizer.generate_sra_lite_url(sra_id, include_download=True)
        print(f"   SRA ID: {sra_id}")
        print(f"   SRA Lite URL: {urls['sra_lite_url']}")
        print(f"   Download URLs: {len([k for k in urls.keys() if 'download' in k])} available")
        
        # Demo 2: Age Extraction
        print("\n2. Age Information Extraction:")
        test_record = {
            'characteristics_ch1': 'Age: P56; Tissue: Brain',
            'source_name_ch1': 'Mouse brain cells from 8-week-old animals'
        }
        age_info = optimizer.extract_age_information(test_record)
        print(f"   Age Value: {age_info['age_value']}")
        print(f"   Age Unit: {age_info['age_unit']}")
        print(f"   Confidence: {age_info['age_confidence']}")
        
        # Demo 3: Tumor Detection
        print("\n3. Tumor Detection:")
        tumor_record = {
            'characteristics_ch1': 'Tissue: Breast cancer tumor',
            'study_title': 'Single-cell analysis of breast carcinoma'
        }
        tumor_info = optimizer.detect_tumor_status(tumor_record)
        print(f"   Is Tumor: {tumor_info['is_tumor']}")
        print(f"   Tumor Type: {tumor_info['tumor_type']}")
        print(f"   Confidence: {tumor_info['confidence']}")
        
        # Demo 4: Cell Line Detection
        print("\n4. Cell Line Detection:")
        cell_line_record = {
            'characteristics_ch1': 'Cell line: HEK293',
            'source_name_ch1': 'HEK293 cell line'
        }
        cell_line_info = optimizer.detect_cell_line(cell_line_record)
        print(f"   Is Cell Line: {cell_line_info['is_cell_line']}")
        print(f"   Cell Line Name: {cell_line_info['cell_line_name']}")
        print(f"   Confidence: {cell_line_info['confidence']}")
        
        # Demo 5: Single-Cell Classification
        print("\n5. Single-Cell Classification:")
        scrna_record = {
            'experiment_title': '10x single-cell RNA sequencing',
            'study_title': 'Single-cell RNA-seq analysis'
        }
        sc_classification = optimizer.classify_single_cell_type(scrna_record)
        print(f"   Experiment Type: {sc_classification['experiment_type']}")
        print(f"   Technology: {sc_classification['technology']}")
        print(f"   Confidence: {sc_classification['confidence']}")
        
        print("\n✅ All individual features demo completed!")
        
    except Exception as e:
        print(f"❌ Error in individual features demo: {e}")

def demo_pmc_analysis():
    """Demonstrate PMC analysis functionality."""
    print("\n📚 PMC Analysis Demo")
    print("-" * 50)
    
    try:
        from scAgent_2025.utils_pmc_analyzer import PmcAnalyzer
        
        pmc_analyzer = PmcAnalyzer()
        
        # Demo text analysis
        print("Analyzing sample text for age and sample size information...")
        sample_text = "We analyzed brain cells from 50 mice aged P56 and 8-week-old animals. The study included 10,000 single cells from human brain tissue."
        
        age_info = pmc_analyzer.extract_age_from_text(sample_text)
        sample_info = pmc_analyzer.extract_sample_size_from_text(sample_text)
        
        print(f"   Age Information: {age_info['age_value']} ({age_info['age_unit']})")
        print(f"   Sample Size: {sample_info['sample_size']} ({sample_info['sample_type']})")
        
        print("✅ PMC analysis demo completed!")
        
    except Exception as e:
        print(f"❌ Error in PMC analysis demo: {e}")

def demo_configuration():
    """Demonstrate configuration options."""
    print("\n⚙️ Configuration Demo")
    print("-" * 50)
    
    try:
        import yaml
        
        # Load enhanced configuration
        config_file = "scAgent_2025/settings_enhanced.yml"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print("Enhanced Configuration Options:")
            print(f"   AI Workers: {config['processing']['max_ai_workers']}")
            print(f"   GPU Enabled: {config['processing']['enable_gpu']}")
            print(f"   Batch Size: {config['processing']['batch_size']}")
            print(f"   Include Downloads: {config['urls']['include_download_urls']}")
            print(f"   PMC Analysis: {config['pmc_analysis']['enable_pmc_analysis']}")
            
            print("✅ Configuration demo completed!")
        else:
            print("⚠️ Enhanced configuration file not found")
            
    except Exception as e:
        print(f"❌ Error in configuration demo: {e}")

def show_usage_examples():
    """Show usage examples."""
    print("\n📖 Usage Examples")
    print("-" * 50)
    
    examples = """
Command Line Examples:

1. Quick test run:
   python enhanced_sc_eqtl_analysis.py --test-run

2. Process 10,000 samples with enhanced features:
   python enhanced_sc_eqtl_analysis.py --quick-scan 10000 --batch-size 2000

3. Full database scan with download URLs:
   python enhanced_sc_eqtl_analysis.py --fullscan --include-downloads

4. High-performance mode with 20 AI workers:
   python enhanced_sc_eqtl_analysis.py --quick-scan 50000 --ai-workers 20

5. Custom output with AI disabled:
   python enhanced_sc_eqtl_analysis.py --output results.csv --no-ai

Python API Examples:

1. Basic usage:
   from enhanced_sc_eqtl_analysis import EnhancedScEqtlAnalysis
   
   analysis = EnhancedScEqtlAnalysis(
       max_ai_workers=20,
       enable_gpu=True,
       include_download_urls=True
   )
   
   result = analysis.run_enhanced_analysis(
       max_samples=10000,
       batch_size=5000,
       output_file="enhanced_results.csv",
       enable_ai=True
   )

2. Individual feature usage:
   from scAgent_2025.utils_enhanced_sc_eqtl_optimizer import EnhancedScEqtlOptimizer
   
   optimizer = EnhancedScEqtlOptimizer()
   
   # Generate SRA Lite URLs
   urls = optimizer.generate_sra_lite_url("ERR4405370", include_download=True)
   
   # Extract age information
   age_info = optimizer.extract_age_information(record)
   
   # Detect tumor status
   tumor_info = optimizer.detect_tumor_status(record)
"""
    
    print(examples)

def main():
    """Main function for quick start demo."""
    print_banner()
    
    print("🚀 Starting Enhanced sc-eQTL Analysis Quick Start Demo")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run demos
    demo_basic_usage()
    demo_individual_features()
    demo_pmc_analysis()
    demo_configuration()
    show_usage_examples()
    
    print("\n" + "=" * 80)
    print("🎉 Enhanced sc-eQTL Analysis Quick Start Demo Completed!")
    print("📚 For detailed documentation, see README_ENHANCED_ANALYSIS.md")
    print("🧪 To run tests, execute: python test_enhanced_features.py")
    print("=" * 80)

if __name__ == "__main__":
    main() 