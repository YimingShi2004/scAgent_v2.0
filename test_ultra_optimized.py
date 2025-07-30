#!/usr/bin/env python3
"""
Ultra-Optimized sc-eQTL Analysis Test Script
Test the performance of ultra-optimized system
"""

import sys
import os
import time
from datetime import datetime

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent_2025'))

def test_ultra_optimized_performance():
    """Test ultra-optimized performance"""
    print("🚀 Testing Ultra-Optimized sc-eQTL Analysis Performance")
    print("=" * 80)
    
    try:
        from enhanced_sc_eqtl_analysis import EnhancedScEqtlAnalysis
        
        # Test configurations
        test_configs = [
            {
                "name": "Small Test (1K samples)",
                "max_samples": 1000,
                "batch_size": 5000,
                "ai_workers": 32,
                "gpu_batch_size": 500
            },
            {
                "name": "Medium Test (10K samples)", 
                "max_samples": 10000,
                "batch_size": 20000,
                "ai_workers": 64,
                "gpu_batch_size": 1000
            },
            {
                "name": "Large Test (50K samples)",
                "max_samples": 50000, 
                "batch_size": 50000,
                "ai_workers": 128,
                "gpu_batch_size": 2000
            }
        ]
        
        for config in test_configs:
            print(f"\n🧪 {config['name']}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Initialize analysis system
            analysis = EnhancedScEqtlAnalysis(
                max_ai_workers=config['ai_workers'],
                enable_gpu=True,
                include_download_urls=True
            )
            
            # Run analysis
            result = analysis.run_enhanced_analysis(
                max_samples=config['max_samples'],
                batch_size=config['batch_size'],
                output_file=f"test_results_{config['max_samples']}.csv",
                enable_ai=True,
                gpu_batch_size=config['gpu_batch_size'],
                auto_download=False
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Output results
            if result["status"] == "completed":
                stats = result["processing_statistics"]
                print(f"✅ Test completed successfully!")
                print(f"   Total Processed: {stats['total_processed']:,}")
                print(f"   Hard Filtered: {stats['hard_filtered']:,}")
                print(f"   Final Passed: {stats['final_passed']:,}")
                print(f"   Filter Time: {stats['filter_time']:.2f}s")
                print(f"   Analysis Time: {stats['analysis_time']:.2f}s")
                print(f"   Total Time: {stats['total_time']:.2f}s")
                print(f"   Records/Second: {stats['records_per_second']:.1f}")
                print(f"   Pass Rate: {(stats['final_passed']/stats['total_processed']*100):.2f}%")
            else:
                print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            
            print(f"   Wall Clock Time: {total_time:.2f}s")
    
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

def test_gpu_acceleration():
    """Test GPU acceleration functionality"""
    print("\n🔬 Testing GPU Acceleration")
    print("-" * 50)
    
    try:
        from scAgent_2025.utils_enhanced_sc_eqtl_optimizer import EnhancedScEqtlOptimizer
        
        # Test GPU initialization
        optimizer = EnhancedScEqtlOptimizer(enable_gpu=True)
        print(f"GPU Device: {optimizer.device}")
        print(f"GPU Enabled: {optimizer.enable_gpu}")
        
        # Test hard filtering
        test_record = {
            'organism_ch1': 'Homo sapiens',
            'experiment_title': '10x single-cell RNA sequencing',
            'characteristics_ch1': 'Brain tissue',
            'library_strategy': 'RNA-Seq',
            'sra_ID': 'TEST123'
        }
        
        passed, exp_type = optimizer._fast_hard_filter(test_record)
        print(f"Test Record Filter Result: {passed}, Type: {exp_type}")
        
        # Test batch filtering
        test_records = [test_record] * 100
        start_time = time.time()
        filtered = optimizer._batch_gpu_filter(test_records)
        filter_time = time.time() - start_time
        
        print(f"Batch Filter Result: {len(filtered)}/{len(test_records)} passed in {filter_time:.3f}s")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

def test_sra_lite_urls():
    """Test SRA Lite URL generation"""
    print("\n🔗 Testing SRA Lite URL Generation")
    print("-" * 50)
    
    try:
        from scAgent_2025.utils_enhanced_sc_eqtl_optimizer import EnhancedScEqtlOptimizer
        
        optimizer = EnhancedScEqtlOptimizer()
        
        test_sra_ids = ["ERR4405370", "SRR1234567", "DRR9876543"]
        
        for sra_id in test_sra_ids:
            urls = optimizer.generate_sra_lite_url(sra_id, include_download=True)
            print(f"SRA ID: {sra_id}")
            print(f"  SRA Lite URL: {urls['sra_lite_url']}")
            print(f"  Download URLs: {len([k for k in urls.keys() if 'download' in k])} available")
        
    except Exception as e:
        print(f"❌ SRA Lite URL test failed: {e}")

def main():
    """Main test function"""
    print("🧪 Ultra-Optimized sc-eQTL Analysis Test Suite")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_gpu_acceleration()
    test_sra_lite_urls()
    test_ultra_optimized_performance()
    
    print("\n" + "=" * 80)
    print("🎉 All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    main() 