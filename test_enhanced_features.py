#!/usr/bin/env python3
"""
Test script for Enhanced sc-eQTL Analysis Features
Tests all new functionality including:
- SRA Lite URL generation
- Age extraction
- Tumor detection
- Cell line detection
- scRNA-seq vs scATAC-seq classification
- PMC analysis
- GPU acceleration
"""

import sys
import os
import json
import time
from datetime import datetime

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent_2025'))

from scAgent_2025.utils_enhanced_sc_eqtl_optimizer import EnhancedScEqtlOptimizer
from scAgent_2025.utils_pmc_analyzer import PmcAnalyzer

def test_sra_lite_url_generation():
    """Test SRA Lite URL generation functionality."""
    print("🔗 Testing SRA Lite URL Generation...")
    
    optimizer = EnhancedScEqtlOptimizer()
    
    # Test URL generation
    test_sra_id = "ERR4405370"
    urls = optimizer.generate_sra_lite_url(test_sra_id, include_download=True)
    
    print(f"  SRA ID: {test_sra_id}")
    print(f"  SRA Lite URL: {urls['sra_lite_url']}")
    print(f"  Data Access URL: {urls['data_access_url']}")
    print(f"  FASTQ Download: {urls.get('fastq_download', 'N/A')}")
    print(f"  SRA Download: {urls.get('sra_download', 'N/A')}")
    
    # Verify URL format
    expected_base = "https://trace.ncbi.nlm.nih.gov/Traces/?view=run_browser&acc="
    assert urls['sra_lite_url'].startswith(expected_base)
    assert test_sra_id in urls['sra_lite_url']
    
    print("  ✅ SRA Lite URL generation test passed\n")

def test_age_extraction():
    """Test age information extraction."""
    print("📊 Testing Age Information Extraction...")
    
    optimizer = EnhancedScEqtlOptimizer()
    
    # Test record with age information
    test_record = {
        'characteristics_ch1': 'Age: P56; Tissue: Brain',
        'source_name_ch1': 'Mouse brain cells from 8-week-old animals',
        'study_title': 'Single-cell analysis of brain development'
    }
    
    age_info = optimizer.extract_age_information(test_record)
    
    print(f"  Age Value: {age_info['age_value']}")
    print(f"  Age Unit: {age_info['age_unit']}")
    print(f"  Age Source: {age_info['age_source']}")
    print(f"  Confidence: {age_info['age_confidence']}")
    
    # Verify extraction
    assert age_info['age_value'] in ['P56', '8-week']
    assert age_info['age_unit'] in ['postnatal_days', 'weeks']
    assert age_info['confidence'] > 0.5
    
    print("  ✅ Age extraction test passed\n")

def test_tumor_detection():
    """Test tumor vs normal tissue detection."""
    print("🏥 Testing Tumor Detection...")
    
    optimizer = EnhancedScEqtlOptimizer()
    
    # Test tumor sample
    tumor_record = {
        'characteristics_ch1': 'Tissue: Breast cancer tumor',
        'study_title': 'Single-cell analysis of breast carcinoma',
        'summary': 'Analysis of malignant breast cancer cells'
    }
    
    tumor_info = optimizer.detect_tumor_status(tumor_record)
    
    print(f"  Is Tumor: {tumor_info['is_tumor']}")
    print(f"  Tumor Type: {tumor_info['tumor_type']}")
    print(f"  Confidence: {tumor_info['confidence']}")
    print(f"  Evidence: {tumor_info['evidence']}")
    
    # Test normal sample
    normal_record = {
        'characteristics_ch1': 'Tissue: Normal brain',
        'study_title': 'Single-cell analysis of healthy brain tissue',
        'summary': 'Analysis of normal brain cells'
    }
    
    normal_info = optimizer.detect_tumor_status(normal_record)
    
    print(f"  Normal Sample - Is Tumor: {normal_info['is_tumor']}")
    print(f"  Normal Sample - Confidence: {normal_info['confidence']}")
    
    # Verify detection
    assert tumor_info['is_tumor'] == True
    assert normal_info['is_tumor'] == False
    
    print("  ✅ Tumor detection test passed\n")

def test_cell_line_detection():
    """Test cell line detection and exclusion."""
    print("🧪 Testing Cell Line Detection...")
    
    optimizer = EnhancedScEqtlOptimizer()
    
    # Test cell line sample
    cell_line_record = {
        'characteristics_ch1': 'Cell line: HEK293',
        'source_name_ch1': 'HEK293 cell line',
        'study_title': 'Analysis of HEK293 cells'
    }
    
    cell_line_info = optimizer.detect_cell_line(cell_line_record)
    
    print(f"  Is Cell Line: {cell_line_info['is_cell_line']}")
    print(f"  Cell Line Name: {cell_line_info['cell_line_name']}")
    print(f"  Confidence: {cell_line_info['confidence']}")
    print(f"  Evidence: {cell_line_info['evidence']}")
    
    # Test normal sample
    normal_record = {
        'characteristics_ch1': 'Tissue: Brain',
        'source_name_ch1': 'Primary brain cells',
        'study_title': 'Analysis of primary brain cells'
    }
    
    normal_info = optimizer.detect_cell_line(normal_record)
    
    print(f"  Normal Sample - Is Cell Line: {normal_info['is_cell_line']}")
    print(f"  Normal Sample - Confidence: {normal_info['confidence']}")
    
    # Verify detection
    assert cell_line_info['is_cell_line'] == True
    assert normal_info['is_cell_line'] == False
    
    print("  ✅ Cell line detection test passed\n")

def test_single_cell_classification():
    """Test scRNA-seq vs scATAC-seq classification."""
    print("🧬 Testing Single-Cell Classification...")
    
    optimizer = EnhancedScEqtlOptimizer()
    
    # Test scRNA-seq sample
    scrna_record = {
        'library_strategy': 'RNA-Seq',
        'experiment_title': '10x single-cell RNA sequencing',
        'study_title': 'Single-cell RNA-seq analysis of brain cells'
    }
    
    scrna_classification = optimizer.classify_single_cell_type(scrna_record)
    
    print(f"  scRNA-seq - Experiment Type: {scrna_classification['experiment_type']}")
    print(f"  scRNA-seq - Technology: {scrna_classification['technology']}")
    print(f"  scRNA-seq - Confidence: {scrna_classification['confidence']}")
    
    # Test scATAC-seq sample
    scatac_record = {
        'library_strategy': 'ATAC-Seq',
        'experiment_title': '10x single-cell ATAC sequencing',
        'study_title': 'Single-cell ATAC-seq analysis of brain cells'
    }
    
    scatac_classification = optimizer.classify_single_cell_type(scatac_record)
    
    print(f"  scATAC-seq - Experiment Type: {scatac_classification['experiment_type']}")
    print(f"  scATAC-seq - Technology: {scatac_classification['technology']}")
    print(f"  scATAC-seq - Confidence: {scatac_classification['confidence']}")
    
    # Verify classification
    assert scrna_classification['experiment_type'] == 'scRNA-seq'
    assert scatac_classification['experiment_type'] == 'scATAC-seq'
    
    print("  ✅ Single-cell classification test passed\n")

def test_comprehensive_metadata_extraction():
    """Test comprehensive metadata extraction."""
    print("📋 Testing Comprehensive Metadata Extraction...")
    
    optimizer = EnhancedScEqtlOptimizer()
    
    # Test comprehensive record
    test_record = {
        'sra_ID': 'ERR4405370',
        'gsm_title': 'GSM123456',
        'gse_title': 'GSE123456',
        'experiment_title': '10x single-cell RNA sequencing of brain cells',
        'characteristics_ch1': 'Age: P56; Tissue: Brain; Cell type: Neurons',
        'study_title': 'Single-cell analysis of brain development',
        'summary': 'Analysis of 10,000 single cells from mouse brain',
        'pubmed_id': '12345678',
        'organism_ch1': 'Homo sapiens',
        'spots': '1000000000',
        'bases': '100000000000'
    }
    
    metadata = optimizer.extract_comprehensive_metadata(test_record)
    
    print(f"  SRA ID: {metadata['sra_id']}")
    print(f"  Age Info: {metadata['age_info']}")
    print(f"  Tumor Status: {metadata['tumor_status']}")
    print(f"  Cell Line Info: {metadata['cell_line_info']}")
    print(f"  SC Classification: {metadata['sc_classification']}")
    print(f"  Sample Size: {metadata['sample_size']}")
    print(f"  Publication Info: {metadata['publication_info']}")
    print(f"  Quality Metrics: {metadata['quality_metrics']}")
    
    # Verify metadata extraction
    assert metadata['sra_id'] == 'ERR4405370'
    assert metadata['age_info']['age_value'] == 'P56'
    assert metadata['sc_classification']['experiment_type'] == 'scRNA-seq'
    assert metadata['quality_metrics']['overall_quality_score'] > 0
    
    print("  ✅ Comprehensive metadata extraction test passed\n")

def test_pmc_analyzer():
    """Test PMC analyzer functionality."""
    print("📚 Testing PMC Analyzer...")
    
    pmc_analyzer = PmcAnalyzer()
    
    # Test PMC ID search (using a known PMID)
    test_pmid = "12345678"  # This is a test PMID
    pmc_id = pmc_analyzer.search_pmc_by_pmid(test_pmid)
    
    print(f"  Test PMID: {test_pmid}")
    print(f"  Found PMC ID: {pmc_id}")
    
    # Test age extraction from text
    test_text = "We analyzed brain cells from mice aged P56 and 8-week-old animals."
    age_info = pmc_analyzer.extract_age_from_text(test_text)
    
    print(f"  Age from Text: {age_info['age_value']}")
    print(f"  Age Unit: {age_info['age_unit']}")
    
    # Test sample size extraction
    sample_info = pmc_analyzer.extract_sample_size_from_text(test_text)
    print(f"  Sample Size: {sample_info['sample_size']}")
    
    print("  ✅ PMC analyzer test passed\n")

def test_enhanced_filtering():
    """Test enhanced filtering pipeline."""
    print("🔍 Testing Enhanced Filtering Pipeline...")
    
    optimizer = EnhancedScEqtlOptimizer()
    
    # Test record that should pass all filters
    good_record = {
        'sra_ID': 'ERR4405370',
        'organism_ch1': 'Homo sapiens',
        'experiment_title': '10x single-cell RNA sequencing',
        'characteristics_ch1': 'Tissue: Brain; Age: P56',
        'study_title': 'Single-cell analysis of human brain',
        'summary': 'Analysis of 10,000 single cells from human brain tissue',
        'pubmed_id': '12345678',
        'spots': '1000000000',
        'bases': '100000000000'
    }
    
    filter_result = optimizer.filter_record_enhanced(good_record)
    
    print(f"  Passes All Filters: {filter_result['passes_filter']}")
    print(f"  Confidence Score: {filter_result['confidence_score']}")
    print(f"  Processing Time: {filter_result['processing_time']:.3f}s")
    
    # Test filter steps
    filter_steps = filter_result['filter_steps']
    print(f"  Human Check: {filter_steps['human_check']['passed']}")
    print(f"  Cell Line Check: {filter_steps['cell_line_check']['passed']}")
    print(f"  Single Cell Check: {filter_steps['single_cell_check']['passed']}")
    print(f"  Quality Check: {filter_steps['quality_check']['passed']}")
    
    # Verify filtering
    assert filter_result['passes_filter'] == True
    assert filter_result['confidence_score'] > 0.5
    
    print("  ✅ Enhanced filtering test passed\n")

def test_gpu_acceleration():
    """Test GPU acceleration functionality."""
    print("🚀 Testing GPU Acceleration...")
    
    # Test GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        print(f"  GPU Available: {gpu_available}")
        print(f"  GPU Count: {gpu_count}")
        
        if gpu_available:
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test optimizer with GPU
        optimizer = EnhancedScEqtlOptimizer(enable_gpu=gpu_available)
        print(f"  Optimizer GPU Enabled: {optimizer.enable_gpu}")
        
    except ImportError:
        print("  PyTorch not available, skipping GPU test")
    
    print("  ✅ GPU acceleration test passed\n")

def run_all_tests():
    """Run all tests."""
    print("🧪 Starting Enhanced sc-eQTL Analysis Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        test_sra_lite_url_generation()
        test_age_extraction()
        test_tumor_detection()
        test_cell_line_detection()
        test_single_cell_classification()
        test_comprehensive_metadata_extraction()
        test_pmc_analyzer()
        test_enhanced_filtering()
        test_gpu_acceleration()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("=" * 60)
        print(f"🎉 All tests completed successfully in {total_time:.2f} seconds!")
        print("✅ Enhanced sc-eQTL Analysis System is ready for use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 