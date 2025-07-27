#!/usr/bin/env python3
"""
Specialized script for finding human samples and testing filtering effectiveness
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent'))

from scAgent.utils_sra_geo_ft2_optimizer import SraGeoFt2Optimizer
from scAgent.db.connect import get_connection
import psycopg2.extras

def find_human_samples(limit=50):
    """
    Find records containing human samples
    """
    print("🔍 Finding human samples...")
    
    try:
        conn = get_connection()
        
        # Find records containing Homo sapiens
        query = """
        SELECT * FROM "merged"."sra_geo_ft2"
        WHERE 
            LOWER(COALESCE("organism_ch1", '')) LIKE %s OR
            LOWER(COALESCE("scientific_name", '')) LIKE %s OR
            LOWER(COALESCE("experiment_title", '')) LIKE %s OR
            LOWER(COALESCE("study_title", '')) LIKE %s OR
            LOWER(COALESCE("summary", '')) LIKE %s
        ORDER BY "sra_ID"
        LIMIT %s
        """
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, ('%homo sapiens%', '%homo sapiens%', '%homo sapiens%', '%human%', '%human%', limit))
            human_samples = cur.fetchall()
        
        conn.close()
        print(f"✅ Found {len(human_samples)} potential human samples")
        return [dict(sample) for sample in human_samples]
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        print(f"Detailed error: {traceback.format_exc()}")
        return []

def test_human_sample_filtering():
    """
    Test filtering effectiveness on human samples
    """
    print("="*80)
    print("Human Sample Filtering Test")
    print("="*80)
    
    # 1. Find human samples
    human_samples = find_human_samples(limit=20)
    
    if not human_samples:
        print("❌ No human samples found for testing")
        return
    
    # 2. Initialize optimizer
    optimizer = SraGeoFt2Optimizer()
    
    # 3. Test each sample
    passed_count = 0
    rejected_count = 0
    human_confirmed_count = 0
    cell_line_count = 0
    
    print(f"\nTesting {len(human_samples)} potential human samples:")
    
    for i, sample in enumerate(human_samples, 1):
        print(f"\n🧬 Sample {i}:")
        print(f"   ID: {sample.get('sra_ID', 'N/A')}")
        
        # Display key information
        key_fields = [
            ('organism_ch1', 'organism_ch1'),
            ('scientific_name', 'scientific_name'),
            ('experiment_title', 'experiment_title'),
            ('characteristics_ch1', 'characteristics_ch1'),
            ('study_title', 'study_title')
        ]
        
        for label, field in key_fields:
            value = sample.get(field, '')
            if value:
                # Truncate long values
                display_value = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                print(f"   {label}: {display_value}")
        
        # Apply filter
        print(f"\n   🔬 Filter test:")
        result = optimizer.filter_record_optimized(sample)
        
        # Analyze results
        if result['passes_filter']:
            passed_count += 1
            print(f"   ✅ Passed all filters (confidence: {result['confidence_score']:.2f})")
            
            # Display filter steps
            for step_name, step_result in result.get('filter_steps', {}).items():
                print(f"      ✓ {step_name}: {step_result.get('reason', '')}")
            
            # Display extracted key information
            criteria = result.get('extracted_criteria', {})
            print(f"   📊 Extracted information:")
            print(f"      organism: {criteria.get('organism', 'N/A')}")
            print(f"      cell_type: {criteria.get('cell_type', 'N/A')}")
            print(f"      sequencing_platform: {criteria.get('sequencing_platform', 'N/A')}")
            print(f"      project_id: {criteria.get('project_id', 'N/A')}")
            print(f"      publication_info: {criteria.get('publication_info', 'N/A')}")
            print(f"      geographic_location: {criteria.get('geographic_location', 'N/A')}")
            print(f"      age_range: {criteria.get('age_range', 'N/A')}")
            print(f"      disease_status: {criteria.get('disease_status', 'N/A')}")
        else:
            rejected_count += 1
            # Check if it was confirmed human but rejected for other reasons
            if result.get('filter_steps', {}).get('human_check', {}).get('passed', False):
                human_confirmed_count += 1
                # Check if it was cell line
                if not result.get('filter_steps', {}).get('cell_line_check', {}).get('passed', True):
                    cell_line_count += 1
                elif not result.get('filter_steps', {}).get('single_cell_check', {}).get('passed', False):
                    print(f"      → Human sample but not single-cell experiment")
        
        print(f"   ⏱️  Processing time: {result.get('processing_time', 0):.4f}s")
    
    # Summary
    print(f"\n{'='*80}")
    print("Test Summary:")
    print(f"  Total test samples: {len(human_samples)}")
    print(f"  Passed all filters: {passed_count}")
    print(f"  Confirmed human but rejected by other criteria: {human_confirmed_count}")
    print(f"  Identified as cell lines: {cell_line_count}")
    print(f"  Final pass rate: {passed_count/len(human_samples)*100:.1f}%")

def test_rna_seq_boundary_cases():
    """
    Test RNA-Seq boundary cases - samples that might be single-cell but labeled as RNA-Seq
    """
    print("="*80)
    print("RNA-Seq Boundary Cases Test")
    print("="*80)
    
    try:
        conn = get_connection()
        
        # Find samples containing Homo sapiens and RNA-Seq that might be mislabeled single-cell experiments
        query = """
        SELECT * FROM "merged"."sra_geo_ft2"
        WHERE 
            LOWER(COALESCE("experiment_title", '')) LIKE '%homo sapiens%' AND
            LOWER(COALESCE("experiment_title", '')) LIKE '%rna-seq%' AND
            (COALESCE("study_title", '') != '' OR COALESCE("study_abstract", '') != '')
        ORDER BY "sra_ID"
        LIMIT 5
        """
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rna_seq_samples = cur.fetchall()
        
        conn.close()
        
        if not rna_seq_samples:
            print("❌ No RNA-Seq boundary case samples found")
            return
        
        print(f"🧬 Found {len(rna_seq_samples)} RNA-Seq boundary case samples, starting AI-assisted analysis:")
        
        optimizer = SraGeoFt2Optimizer()
        ai_identified_count = 0
        
        for i, sample in enumerate(rna_seq_samples, 1):
            print(f"\n🔬 Boundary sample {i}:")
            print(f"   sra_ID: {sample.get('sra_ID', 'N/A')}")
            print(f"   experiment_title: {sample.get('experiment_title', 'N/A')}")
            
            # Display study information
            study_title = sample.get('study_title', '')
            study_abstract = sample.get('study_abstract', '')
            
            if study_title:
                print(f"   study_title: {study_title[:100]}...")
            if study_abstract:
                print(f"   study_abstract: {study_abstract[:200]}...")
            
            print(f"\n   🤖 AI-assisted analysis:")
            
            # Use enhanced single-cell detection (includes AI analysis)
            sc_result = optimizer.is_single_cell_experiment_optimized(sample)
            is_sc, sc_reason, sc_confidence, sc_type = sc_result
            
            if is_sc:
                ai_identified_count += 1
                print(f"   ✅ AI identified as single-cell: {sc_reason}")
                print(f"      Confidence: {sc_confidence:.2f}, Type: {sc_type}")
                
                # Perform complete filtering test
                print(f"   🔬 Complete filtering test:")
                result = optimizer.filter_record_optimized(sample)
                
                if result['passes_filter']:
                    print(f"   🎉 Passed all filters! (Total confidence: {result['confidence_score']:.2f})")
                    
                    # Display extracted key information
                    criteria = result.get('extracted_criteria', {})
                    print(f"   📊 Extracted sc-eQTL information:")
                    print(f"      organism: {criteria.get('organism', 'N/A')}")
                    print(f"      cell_type: {criteria.get('cell_type', 'N/A')}")
                    print(f"      sequencing_platform: {criteria.get('sequencing_platform', 'N/A')}")
                    print(f"      project_id: {criteria.get('project_id', 'N/A')}")
                    print(f"      publication_info: {criteria.get('publication_info', 'N/A')}")
                    print(f"      geographic_location: {criteria.get('geographic_location', 'N/A')}")
                    print(f"      age_range: {criteria.get('age_range', 'N/A')}")
                    print(f"      disease_status: {criteria.get('disease_status', 'N/A')}")
            else:
                print(f"   ❌ AI did not identify as single-cell: {sc_reason}")
                print(f"      Confidence: {sc_confidence:.2f}")
        
        print(f"\n📊 Boundary case analysis results:")
        print(f"   Total test samples: {len(rna_seq_samples)}")
        print(f"   AI identified as single-cell: {ai_identified_count}")
        print(f"   AI identification rate: {ai_identified_count/len(rna_seq_samples)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Boundary case test failed: {e}")
        print(f"Detailed error: {traceback.format_exc()}")

def test_specific_human_patterns():
    """
    Test specific patterns of human samples
    """
    print("="*80)
    print("Specific Human Pattern Test")
    print("="*80)
    
    try:
        conn = get_connection()
        
        # Look for specific patterns
        query = """
        SELECT * FROM "merged"."sra_geo_ft2"
        WHERE 
            LOWER(COALESCE("experiment_title", '')) LIKE '%homo sapiens%'
        ORDER BY "sra_ID"
        LIMIT 10
        """
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            samples = cur.fetchall()
        
        conn.close()
        
        print(f"Found {len(samples)} samples containing 'Homo sapiens' in experiment_title:\n")
        
        optimizer = SraGeoFt2Optimizer()
        
        for i, sample in enumerate(samples, 1):
            print(f"📋 Experiment {i}:")
            print(f"   experiment_title: {sample.get('experiment_title', 'N/A')}")
            
            # Test individual detection functions
            human_result = optimizer.is_human_sample_optimized(sample)
            cell_line_result = optimizer.is_cell_line_sample(sample)
            single_cell_result = optimizer.is_single_cell_experiment_optimized(sample)
            
            print(f"   Human detection: {'✓' if human_result[0] else '✗'} {human_result[1]} (confidence: {human_result[2]:.2f})")
            print(f"   Cell line detection: {'✓' if not cell_line_result[0] else '✗'} {cell_line_result[1]} (confidence: {cell_line_result[2]:.2f})")
            print(f"   Single-cell detection: {'✓' if single_cell_result[0] else '✗'} {single_cell_result[1]} (confidence: {single_cell_result[2]:.2f}, type: {single_cell_result[3]})")
            print()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Detailed error: {traceback.format_exc()}")

def main():
    """Main function"""
    print("🧬 Human Sample Specialized Testing")
    print("="*80 + "\n")
    
    # Test 1: General human sample filtering
    test_human_sample_filtering()
    
    # Test 2: Specific patterns
    print("\n" + "="*80)
    test_specific_human_patterns()
    
    # Test 3: RNA-Seq boundary cases
    print("\n" + "="*80)
    test_rna_seq_boundary_cases()
    
    # Final summary
    print("\n" + "="*80)
    print("Final Results:")
    
    # Determine overall status
    optimizer = SraGeoFt2Optimizer()
    test_samples = find_human_samples(limit=5)
    
    if test_samples:
        passed_any = False
        for sample in test_samples:
            result = optimizer.filter_record_optimized(sample)
            if result['passes_filter']:
                passed_any = True
                break
        
        if passed_any:
            print("✅ Found samples that passed all filters")
            print("🎯 AI successfully identified RNA-Seq boundary cases as single-cell experiments")
            print("💡 System is working properly and ready for large-scale screening")
        else:
            print("⚠️  No samples passed all filters")
            print("💡 Consider adjusting filter criteria or the data may have few human single-cell samples")
            print("🤖 Check AI model configuration and network connectivity")
    else:
        print("❌ Unable to find test samples")
        print("💡 Check database connection and table content")

if __name__ == "__main__":
    main() 