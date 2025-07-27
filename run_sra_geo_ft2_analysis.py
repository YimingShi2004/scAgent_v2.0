#!/usr/bin/env python3
"""
Production script for running sra_geo_ft2 analysis with optimized filtering.
专门针对 sra_geo_ft2 表的生产级分析脚本。
"""

import sys
import os
import argparse
from datetime import datetime

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent'))

from scAgent.utils_sra_geo_ft2_optimizer import SraGeoFt2Optimizer

def main():
    parser = argparse.ArgumentParser(description='SRA-GEO-FT2 优化分析工具')
    parser.add_argument('--batch-size', type=int, default=10000, 
                       help='批处理大小 (默认: 10000)')
    parser.add_argument('--max-records', type=int, default=None,
                       help='最大处理记录数 (默认: 全部)')
    parser.add_argument('--output', type=str, 
                       default=f'sra_geo_ft2_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                       help='输出文件路径')
    parser.add_argument('--parallel', action='store_true',
                       help='启用并行处理')
    parser.add_argument('--preview-only', action='store_true',
                       help='仅预览表格结构，不执行过滤')
    
    args = parser.parse_args()
    
    print("SRA-GEO-FT2 优化分析工具")
    print("=" * 50)
    
    try:
        # 初始化优化器
        optimizer = SraGeoFt2Optimizer()
        
        if args.preview_only:
            # 仅预览模式
            print("获取表格预览...")
            preview = optimizer.get_table_preview(limit=5)
            
            if "error" not in preview:
                info = preview['table_info']
                print(f"表格: {info['schema']}.{info['table']}")
                print(f"总记录数: {info['total_records']:,}")
                print(f"总列数: {info['total_columns']}")
                
                print("\n关键列分类:")
                for category, columns in preview['relevant_columns'].items():
                    print(f"  {category}: {len(columns)} 列")
                
                print(f"\n样本数据预览:")
                for i, sample in enumerate(preview['sample_data'], 1):
                    print(f"样本 {i}:")
                    print(f"  run_accession: {sample.get('run_accession', 'N/A')}")
                    print(f"  organism_ch1: {sample.get('organism_ch1', 'N/A')}")
                    experiment_title = sample.get('experiment_title', 'N/A')
                    if len(str(experiment_title)) > 80:
                        experiment_title = str(experiment_title)[:80] + "..."
                    print(f"  experiment_title: {experiment_title}")
                    print()
            else:
                print(f"预览失败: {preview['error']}")
                return 1
        else:
            # 执行过滤分析
            print(f"开始批量过滤分析...")
            print(f"批处理大小: {args.batch_size:,}")
            if args.max_records:
                print(f"最大处理记录: {args.max_records:,}")
            else:
                print("处理所有记录")
            print(f"并行处理: {'启用' if args.parallel else '禁用'}")
            print(f"输出文件: {args.output}")
            print("-" * 50)
            
            # 执行批量过滤
            results = optimizer.batch_filter_optimized(
                batch_size=args.batch_size,
                max_records=args.max_records,
                output_file=args.output,
                enable_parallel=args.parallel
            )
            
            # 显示结果
            if "error" not in results.get("processing_summary", {}):
                summary = results["processing_summary"]
                print("\n分析完成!")
                print("=" * 50)
                print(f"总处理记录: {summary['total_processed']:,}")
                print(f"通过记录: {summary['total_passed']:,}")
                print(f"通过率: {summary['pass_rate']:.2f}%")
                print(f"处理时间: {summary['processing_time']:.2f} 秒")
                print(f"处理速度: {summary['records_per_second']:.0f} 记录/秒")
                
                # 过滤统计
                if "filter_statistics" in results:
                    print(f"\n过滤统计:")
                    stats = results["filter_statistics"]
                    print(f"  总处理数: {stats.get('total_processed', 0)}")
                    print(f"  通过过滤数: {stats.get('passed_filters', 0)}")
                    
                    # 单细胞类型统计
                    sc_types = [k for k in stats.keys() if k.startswith('sc_type_')]
                    if sc_types:
                        print(f"  单细胞类型分布:")
                        for sc_type in sc_types:
                            type_name = sc_type.replace('sc_type_', '')
                            print(f"    {type_name}: {stats[sc_type]}")
                
                # 通过的样本示例
                if results.get("results"):
                    print(f"\n通过过滤的样本示例 (前5个):")
                    for i, result in enumerate(results["results"][:5], 1):
                        print(f"{i}. {result['record_id']} (置信度: {result['confidence_score']:.2f})")
                        
                        # 显示提取的关键信息
                        criteria = result.get('extracted_criteria', {})
                        key_info = []
                        if criteria.get('organism'):
                            key_info.append(f"物种: {criteria['organism']}")
                        if criteria.get('tissue_type'):
                            key_info.append(f"组织: {criteria['tissue_type']}")
                        if criteria.get('sequencing_platform'):
                            key_info.append(f"平台: {criteria['sequencing_platform']}")
                        if criteria.get('sample_size'):
                            key_info.append(f"样本量: {criteria['sample_size']}")
                        
                        if key_info:
                            print(f"   {' | '.join(key_info)}")
                
                print(f"\n结果已保存到: {args.output}")
                return 0
            else:
                error = results["processing_summary"].get("error", "Unknown error")
                print(f"分析失败: {error}")
                return 1
                
    except Exception as e:
        print(f"程序异常: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 