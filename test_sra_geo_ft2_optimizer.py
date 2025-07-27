#!/usr/bin/env python3
"""
Test script for the optimized sra_geo_ft2 filtering system.
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent'))

from scAgent.utils_sra_geo_ft2_optimizer import SraGeoFt2Optimizer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'sra_geo_ft2_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def test_table_preview():
    """Test table preview functionality."""
    print("=" * 60)
    print("测试表格预览功能 (Table Preview Test)")
    print("=" * 60)
    
    try:
        optimizer = SraGeoFt2Optimizer()
        preview = optimizer.get_table_preview(limit=3)
        
        if "error" not in preview:
            table_info = preview['table_info']
            print(f"✓ 表格信息: {table_info['schema']}.{table_info['table']}")
            print(f"✓ 总记录数: {table_info['total_records']:,}")
            print(f"✓ 总列数: {table_info['total_columns']}")
            
            print(f"\n关键列识别:")
            for category, columns in preview['relevant_columns'].items():
                print(f"  - {category}: {columns}")
            
            print(f"\n样本数据 (前3条):")
            for i, sample in enumerate(preview['sample_data'], 1):
                print(f"  样本 {i}:")
                # 只显示关键字段 (基于实际表结构)
                key_fields = ['sra_ID', 'organism_ch1', 'scientific_name', 'experiment_title', 'characteristics_ch1', 'study_title']
                for field in key_fields:
                    value = sample.get(field, 'N/A')
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"    {field}: {value}")
                print()
            
            return True
        else:
            print(f"✗ 预览失败: {preview['error']}")
            return False
            
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        return False

def test_single_record_filtering():
    """Test filtering on individual records."""
    print("=" * 60)
    print("测试单条记录过滤 (Single Record Filtering Test)")
    print("=" * 60)
    
    try:
        optimizer = SraGeoFt2Optimizer()
        
        # Get sample records first
        preview = optimizer.get_table_preview(limit=10)
        if "error" in preview:
            print(f"✗ 无法获取样本数据: {preview['error']}")
            return False
        
        sample_records = preview['sample_data']
        print(f"获取到 {len(sample_records)} 条样本记录进行测试")
        
        passed_count = 0
        for i, record in enumerate(sample_records, 1):
            print(f"\n--- 测试记录 {i} ---")
            record_id = record.get('run_accession', f'record_{i}')
            print(f"记录ID: {record_id}")
            
            # 显示关键字段
            print(f"sra_ID: {record.get('sra_ID', 'NULL')}")
            print(f"organism_ch1: {record.get('organism_ch1', 'NULL')}")
            print(f"scientific_name: {record.get('scientific_name', 'NULL')}")
            experiment_title = record.get('experiment_title', 'NULL')
            if len(str(experiment_title)) > 80:
                experiment_title = str(experiment_title)[:80] + "..."
            print(f"experiment_title: {experiment_title}")
            print(f"characteristics_ch1: {str(record.get('characteristics_ch1', 'NULL'))[:60]}...")
            
            # 应用过滤
            result = optimizer.filter_record_optimized(record)
            
            print(f"过滤结果: {'通过' if result['passes_filter'] else '被拒绝'}")
            print(f"置信度: {result['confidence_score']:.2f}")
            print(f"处理时间: {result['processing_time']:.4f}s")
            
            if result['passes_filter']:
                passed_count += 1
                print("过滤步骤:")
                for step, step_result in result['filter_steps'].items():
                    status = "✓" if step_result['passed'] else "✗"
                    print(f"  {status} {step}: {step_result['reason']}")
                
                print("提取的条件:")
                criteria = result['extracted_criteria']
                for key, value in criteria.items():
                    if value and value != "Not specified":
                        print(f"  {key}: {value}")
            else:
                print(f"拒绝原因: {result['rejection_reason']}")
        
        print(f"\n总结: {passed_count}/{len(sample_records)} 条记录通过过滤")
        return True
        
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        return False

def test_batch_filtering():
    """Test batch filtering functionality."""
    print("=" * 60)
    print("测试批量过滤 (Batch Filtering Test)")
    print("=" * 60)
    
    try:
        optimizer = SraGeoFt2Optimizer()
        
        # 测试小批量处理
        print("开始小批量测试 (5000条记录)...")
        results = optimizer.batch_filter_optimized(
            batch_size=1000,
            max_records=5000,
            enable_parallel=False,
            output_file=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if "error" not in results.get("processing_summary", {}):
            summary = results["processing_summary"]
            print(f"\n批量处理结果:")
            print(f"✓ 总处理记录: {summary['total_processed']:,}")
            print(f"✓ 通过记录: {summary['total_passed']:,}")
            print(f"✓ 通过率: {summary['pass_rate']:.2f}%")
            print(f"✓ 处理时间: {summary['processing_time']:.2f}s")
            print(f"✓ 处理速度: {summary['records_per_second']:.0f} 记录/秒")
            
            # 显示过滤统计
            if "filter_statistics" in results:
                print(f"\n过滤统计:")
                stats = results["filter_statistics"]
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            # 显示通过的样本
            if results["results"]:
                print(f"\n通过过滤的样本 (前3个):")
                for i, result in enumerate(results["results"][:3], 1):
                    print(f"  {i}. {result['record_id']} (置信度: {result['confidence_score']:.2f})")
                    criteria = result.get('extracted_criteria', {})
                    if criteria.get('organism'):
                        print(f"     物种: {criteria['organism']}")
                    if criteria.get('tissue_type'):
                        print(f"     组织: {criteria['tissue_type']}")
                    if criteria.get('sequencing_platform'):
                        print(f"     平台: {criteria['sequencing_platform']}")
            
            return True
        else:
            print(f"✗ 批量处理失败: {results['processing_summary'].get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        return False

def main():
    """Main test function."""
    print("SRA-GEO-FT2 优化器测试")
    print("=" * 60)
    
    setup_logging()
    
    test_results = {
        "table_preview": False,
        "single_record": False,
        "batch_filtering": False
    }
    
    # 测试1: 表格预览
    test_results["table_preview"] = test_table_preview()
    
    # 测试2: 单条记录过滤
    if test_results["table_preview"]:
        test_results["single_record"] = test_single_record_filtering()
    
    # 测试3: 批量过滤
    if test_results["single_record"]:
        test_results["batch_filtering"] = test_batch_filtering()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结 (Test Summary)")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    if all_passed:
        print(f"\n🎉 所有测试通过! SRA-GEO-FT2 优化器工作正常")
        return 0
    else:
        print(f"\n⚠️  部分测试失败，请检查日志")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 