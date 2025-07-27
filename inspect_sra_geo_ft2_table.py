#!/usr/bin/env python3
"""
专门检查 sra_geo_ft2 表结构的脚本
Inspect sra_geo_ft2 table structure and generate column mapping
"""

import sys
import os
import json
from typing import Dict, List, Any

# Add scAgent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scAgent'))

from scAgent.db.connect import get_connection
import psycopg2.extras

def inspect_sra_geo_ft2_table():
    """
    详细检查 sra_geo_ft2 表的结构和内容
    """
    print("=" * 70)
    print("检查 merged.sra_geo_ft2 表结构")
    print("=" * 70)
    
    try:
        conn = get_connection()
        
        # 1. 获取表的基本信息
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 检查表是否存在
            cur.execute("""
                SELECT COUNT(*) as exists
                FROM information_schema.tables 
                WHERE table_schema = 'merged' AND table_name = 'sra_geo_ft2'
            """)
            table_exists = cur.fetchone()['exists']
            
            if not table_exists:
                print("❌ 表 merged.sra_geo_ft2 不存在!")
                return None
            
            print("✅ 表 merged.sra_geo_ft2 存在")
            
            # 获取行数
            cur.execute('SELECT COUNT(*) as total_rows FROM "merged"."sra_geo_ft2"')
            total_rows = cur.fetchone()['total_rows']
            print(f"📊 总行数: {total_rows:,}")
            
            # 2. 获取所有列的详细信息
            cur.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    ordinal_position
                FROM information_schema.columns 
                WHERE table_schema = 'merged' AND table_name = 'sra_geo_ft2'
                ORDER BY ordinal_position
            """)
            
            columns = cur.fetchall()
            print(f"📋 总列数: {len(columns)}")
            print()
            
            # 3. 显示所有列的信息
            print("列详细信息:")
            print("-" * 70)
            print(f"{'序号':<4} {'列名':<25} {'数据类型':<15} {'可空':<5} {'字符长度':<8}")
            print("-" * 70)
            
            for col in columns:
                seq = col['ordinal_position']
                name = col['column_name']
                dtype = col['data_type']
                nullable = 'YES' if col['is_nullable'] == 'YES' else 'NO'
                max_len = col['character_maximum_length'] or '-'
                
                print(f"{seq:<4} {name:<25} {dtype:<15} {nullable:<5} {max_len:<8}")
            
            # 4. 识别关键列并分类
            key_columns = categorize_columns([col['column_name'] for col in columns])
            
            print("\n" + "=" * 70)
            print("关键列分类:")
            print("=" * 70)
            for category, cols in key_columns.items():
                if cols:
                    print(f"\n🔍 {category.upper()} ({len(cols)} 列):")
                    for col in cols:
                        print(f"   - {col}")
            
            # 5. 获取样本数据
            print("\n" + "=" * 70)
            print("样本数据分析 (前3行):")
            print("=" * 70)
            
            cur.execute('SELECT * FROM "merged"."sra_geo_ft2" LIMIT 3')
            sample_data = cur.fetchall()
            
            for i, row in enumerate(sample_data, 1):
                print(f"\n📄 样本 {i}:")
                # 显示关键字段
                key_fields = [
                    'run_accession', 'study_accession', 'sample_accession', 'experiment_accession',
                    'organism_ch1', 'organism', 'experiment_title', 'characteristics_ch1',
                    'study_title', 'library_strategy', 'instrument_model', 'platform'
                ]
                
                for field in key_fields:
                    if field in row:
                        value = row[field]
                        if value is not None:
                            if isinstance(value, str) and len(value) > 60:
                                value = value[:60] + "..."
                            print(f"   {field}: {value}")
                        else:
                            print(f"   {field}: NULL")
            
            # 6. 生成列映射配置
            column_mapping = generate_column_mapping(columns)
            
            print("\n" + "=" * 70)
            print("生成的列映射配置:")
            print("=" * 70)
            print(json.dumps(column_mapping, indent=2, ensure_ascii=False))
            
            # 7. 保存映射到文件
            with open('sra_geo_ft2_column_mapping.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'table_info': {
                        'schema': 'merged',
                        'table': 'sra_geo_ft2',
                        'total_rows': total_rows,
                        'total_columns': len(columns)
                    },
                    'columns': [dict(col) for col in columns],
                    'column_mapping': column_mapping,
                    'key_columns': key_columns
                }, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 列映射信息已保存到: sra_geo_ft2_column_mapping.json")
            
            conn.close()
            return column_mapping
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return None

def categorize_columns(column_names: List[str]) -> Dict[str, List[str]]:
    """
    根据列名特征对列进行分类
    """
    categories = {
        'identifiers': [],      # 标识符列
        'organism_info': [],    # 物种信息列
        'sample_info': [],      # 样本信息列
        'experiment_info': [],  # 实验信息列
        'sequencing_info': [],  # 测序信息列
        'study_info': [],       # 研究信息列
        'quality_info': [],     # 质量信息列
        'date_info': [],        # 日期信息列
        'location_info': [],    # 位置信息列
        'other': []             # 其他列
    }
    
    # 定义分类规则
    patterns = {
        'identifiers': [
            'accession', 'id', 'identifier', 'gsm', 'srr', 'srx', 'srs', 'srp', 'gse', 'geo'
        ],
        'organism_info': [
            'organism', 'species', 'scientific_name', 'taxon', 'strain', 'isolate'
        ],
        'sample_info': [
            'sample', 'source', 'characteristics', 'tissue', 'cell', 'organ', 'body_part'
        ],
        'experiment_info': [
            'experiment', 'title', 'description', 'protocol', 'treatment', 'condition'
        ],
        'sequencing_info': [
            'library', 'platform', 'instrument', 'strategy', 'source', 'selection', 
            'layout', 'spots', 'bases', 'avgspotlen'
        ],
        'study_info': [
            'study', 'project', 'abstract', 'summary', 'publication', 'pubmed'
        ],
        'quality_info': [
            'quality', 'score', 'filter', 'pass', 'fail', 'qc'
        ],
        'date_info': [
            'date', 'time', 'created', 'updated', 'submitted', 'released'
        ],
        'location_info': [
            'country', 'location', 'geographic', 'region', 'center', 'lab'
        ]
    }
    
    # 对每个列名进行分类
    for col_name in column_names:
        col_lower = col_name.lower()
        categorized = False
        
        for category, keywords in patterns.items():
            if any(keyword in col_lower for keyword in keywords):
                categories[category].append(col_name)
                categorized = True
                break
        
        if not categorized:
            categories['other'].append(col_name)
    
    return categories

def generate_column_mapping(columns) -> Dict[str, str]:
    """
    为 sc-eQTL 分析生成关键列的映射
    """
    column_names = [col['column_name'] for col in columns]
    
    # 定义需要映射的关键字段
    mapping_rules = {
        # 标识符字段
        'run_accession': ['run_accession', 'sra_run_accession', 'run_id'],
        'study_accession': ['study_accession', 'sra_study_accession', 'study_id', 'project_accession'],
        'sample_accession': ['sample_accession', 'sra_sample_accession', 'sample_id'],
        'experiment_accession': ['experiment_accession', 'sra_experiment_accession', 'experiment_id'],
        'geo_accession': ['geo_accession', 'gse', 'geo_series'],
        
        # 物种相关字段
        'organism': ['organism', 'species'],
        'organism_ch1': ['organism_ch1', 'organism_1'],
        'scientific_name': ['scientific_name', 'organism_name'],
        'taxon_id': ['taxon_id', 'tax_id', 'taxonomy_id'],
        
        # 实验信息字段
        'experiment_title': ['experiment_title', 'title', 'gsm_title'],
        'study_title': ['study_title', 'project_title'],
        'study_abstract': ['study_abstract', 'abstract', 'summary'],
        
        # 样本信息字段
        'sample_title': ['sample_title', 'sample_name'],
        'source_name': ['source_name', 'source'],
        'characteristics_ch1': ['characteristics_ch1', 'characteristics', 'sample_characteristics'],
        'tissue': ['tissue', 'tissue_type', 'organ'],
        
        # 测序信息字段
        'library_strategy': ['library_strategy', 'strategy'],
        'library_source': ['library_source', 'source'],
        'library_selection': ['library_selection', 'selection'],
        'library_layout': ['library_layout', 'layout'],
        'platform': ['platform', 'sequencing_platform'],
        'instrument_model': ['instrument_model', 'instrument', 'machine'],
        
        # 数据量字段
        'spots': ['spots', 'spot_count', 'read_count'],
        'bases': ['bases', 'base_count'],
        'avgspotlen': ['avgspotlen', 'avg_spot_length', 'read_length'],
        
        # 日期字段
        'submission_date': ['submission_date', 'submitted_date', 'create_date'],
        'publication_date': ['publication_date', 'release_date', 'public_date'],
        
        # 其他重要字段
        'bioproject': ['bioproject', 'project_id'],
        'biosample': ['biosample', 'sample_id'],
        'center_name': ['center_name', 'center', 'submitter'],
        'description': ['description', 'desc']
    }
    
    # 执行映射
    column_mapping = {}
    for logical_name, possible_names in mapping_rules.items():
        for possible_name in possible_names:
            # 精确匹配
            if possible_name in column_names:
                column_mapping[logical_name] = possible_name
                break
            # 模糊匹配
            for col_name in column_names:
                if possible_name.lower() in col_name.lower():
                    column_mapping[logical_name] = col_name
                    break
            if logical_name in column_mapping:
                break
    
    return column_mapping

def main():
    """主函数"""
    print("🔍 SRA-GEO-FT2 表结构检查工具")
    
    mapping = inspect_sra_geo_ft2_table()
    
    if mapping:
        print("\n✅ 检查完成！可以使用生成的列映射更新优化器。")
        return 0
    else:
        print("\n❌ 检查失败！")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 