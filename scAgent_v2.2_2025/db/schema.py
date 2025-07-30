"""
Database schema analysis utilities for scAgent.
"""

import psycopg2
import psycopg2.extras
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
from .connect import get_connection, get_cursor
from datetime import datetime

logger = logging.getLogger(__name__)

def analyze_table_schema(
    table_name: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, Any]:
    """
    Analyze the schema of a specific table.
    
    Args:
        table_name: Name of the table to analyze
        conn: Database connection (optional, will create if not provided)
        
    Returns:
        Dict containing table schema information
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        with get_cursor(conn, psycopg2.extras.RealDictCursor) as cur:
            # Get table columns information
            cur.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    ordinal_position
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND table_schema = 'public'
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = cur.fetchall()
            
            # Get table statistics
            cur.execute(f"""
                SELECT 
                    COUNT(*) as row_count,
                    pg_size_pretty(pg_total_relation_size('{table_name}')) as table_size
                FROM {table_name};
            """)
            
            stats = cur.fetchone()
            
            # Get sample data (first 5 rows)
            cur.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            sample_data = cur.fetchall()
            
            # Get indexes
            cur.execute("""
                SELECT 
                    indexname,
                    indexdef
                FROM pg_indexes 
                WHERE tablename = %s 
                AND schemaname = 'public';
            """, (table_name,))
            
            indexes = cur.fetchall()
            
            # Get foreign keys
            cur.execute("""
                SELECT
                    tc.constraint_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = %s
                AND tc.table_schema = 'public';
            """, (table_name,))
            
            foreign_keys = cur.fetchall()
            
        return {
            "table_name": table_name,
            "columns": [dict(col) for col in columns],
            "row_count": stats["row_count"],
            "table_size": stats["table_size"],
            "sample_data": [dict(row) for row in sample_data],
            "indexes": [dict(idx) for idx in indexes],
            "foreign_keys": [dict(fk) for fk in foreign_keys],
            "column_count": len(columns)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing table schema for {table_name}: {e}")
        raise
    finally:
        if should_close:
            conn.close()

def get_table_info(
    table_names: List[str] = None,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get information about multiple tables.
    
    Args:
        table_names: List of table names to analyze (defaults to geo_master, sra_master)
        conn: Database connection (optional)
        
    Returns:
        Dict mapping table names to their schema information
    """
    if table_names is None:
        table_names = ["geo_master", "sra_master"]
    
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        table_info = {}
        for table_name in table_names:
            try:
                table_info[table_name] = analyze_table_schema(table_name, conn)
                logger.info(f"Successfully analyzed schema for table: {table_name}")
            except Exception as e:
                logger.error(f"Failed to analyze table {table_name}: {e}")
                table_info[table_name] = {"error": str(e)}
                
        return table_info
        
    finally:
        if should_close:
            conn.close()

def get_column_statistics(
    table_name: str,
    column_name: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, Any]:
    """
    Get detailed statistics for a specific column.
    
    Args:
        table_name: Name of the table
        column_name: Name of the column
        conn: Database connection (optional)
        
    Returns:
        Dict containing column statistics
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        with get_cursor(conn, psycopg2.extras.RealDictCursor) as cur:
            # Get basic statistics
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT({column_name}) as non_null_count,
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    COUNT(*) - COUNT({column_name}) as null_count
                FROM {table_name};
            """)
            
            basic_stats = cur.fetchone()
            
            # Get most common values
            cur.execute(f"""
                SELECT 
                    {column_name},
                    COUNT(*) as frequency
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                GROUP BY {column_name}
                ORDER BY frequency DESC
                LIMIT 10;
            """)
            
            common_values = cur.fetchall()
            
            # Calculate null percentage
            null_percentage = (basic_stats["null_count"] / basic_stats["total_count"]) * 100
            
            return {
                "table_name": table_name,
                "column_name": column_name,
                "total_count": basic_stats["total_count"],
                "non_null_count": basic_stats["non_null_count"],
                "null_count": basic_stats["null_count"],
                "null_percentage": round(null_percentage, 2),
                "distinct_count": basic_stats["distinct_count"],
                "common_values": [dict(cv) for cv in common_values]
            }
            
    except Exception as e:
        logger.error(f"Error getting column statistics for {table_name}.{column_name}: {e}")
        raise
    finally:
        if should_close:
            conn.close()

def detect_potential_eqtl_columns(
    table_name: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, List[str]]:
    """
    Detect columns that might be relevant for sc-eQTL analysis.
    
    Args:
        table_name: Name of the table to analyze
        conn: Database connection (optional)
        
    Returns:
        Dict categorizing columns by potential relevance to sc-eQTL
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        schema_info = analyze_table_schema(table_name, conn)
        columns = [col["column_name"].lower() for col in schema_info["columns"]]
        
        # Define patterns for different types of relevant columns
        eqtl_relevant = {
            "organism": [],
            "tissue": [],
            "cell_type": [],
            "individual": [],
            "sequencing": [],
            "quality": [],
            "metadata": [],
            "identifiers": []
        }
        
        # Patterns to match
        patterns = {
            "organism": ["organism", "species", "taxon", "scientific_name"],
            "tissue": ["tissue", "organ", "anatomical", "body_part"],
            "cell_type": ["cell_type", "cell_line", "celltype", "cell"],
            "individual": ["individual", "subject", "patient", "donor", "sample_id"],
            "sequencing": ["seq", "rna", "assay", "platform", "instrument", "library"],
            "quality": ["quality", "score", "qc", "filter", "pass", "fail"],
            "metadata": ["title", "description", "summary", "abstract", "characteristic"],
            "identifiers": ["accession", "id", "gsm", "srx", "srr", "geo"]
        }
        
        # Categorize columns
        for column in columns:
            for category, pattern_list in patterns.items():
                if any(pattern in column for pattern in pattern_list):
                    eqtl_relevant[category].append(column)
                    break
        
        return eqtl_relevant
        
    finally:
        if should_close:
            conn.close()

def export_schema_report(
    table_names: List[str] = None,
    output_file: str = "schema_report.txt"
) -> str:
    """
    Export a comprehensive schema report to a file.
    
    Args:
        table_names: List of table names to analyze
        output_file: Output file path
        
    Returns:
        Path to the generated report file
    """
    if table_names is None:
        table_names = ["geo_master", "sra_master"]
    
    table_info = get_table_info(table_names)
    
    with open(output_file, 'w') as f:
        f.write("scAgent Database Schema Report\n")
        f.write("=" * 50 + "\n\n")
        
        for table_name, info in table_info.items():
            if "error" in info:
                f.write(f"Table: {table_name}\n")
                f.write(f"Error: {info['error']}\n\n")
                continue
                
            f.write(f"Table: {table_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Row Count: {info['row_count']:,}\n")
            f.write(f"Table Size: {info['table_size']}\n")
            f.write(f"Column Count: {info['column_count']}\n\n")
            
            f.write("Columns:\n")
            for col in info['columns']:
                f.write(f"  - {col['column_name']}: {col['data_type']}")
                if col['is_nullable'] == 'NO':
                    f.write(" (NOT NULL)")
                f.write("\n")
            
            f.write("\nIndexes:\n")
            for idx in info['indexes']:
                f.write(f"  - {idx['indexname']}\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    logger.info(f"Schema report exported to: {output_file}")
    return output_file

def analyze_column_details(
    table_name: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, Any]:
    """
    Analyze detailed column information including data distribution and unique values.
    
    Args:
        table_name: Name of the table to analyze
        conn: Database connection (optional)
        
    Returns:
        Detailed column analysis
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        cursor = conn.cursor()
        
        # Get basic schema info
        schema_info = analyze_table_schema(table_name, conn)
        
        column_details = {}
        
        for col_info in schema_info["columns"]:
            col_name = col_info["column_name"]
            col_type = col_info["data_type"]
            
            # Initialize column analysis
            column_analysis = {
                "column_name": col_name,
                "data_type": col_type,
                "is_nullable": col_info["is_nullable"],
                "total_count": 0,
                "null_count": 0,
                "unique_count": 0,
                "sample_values": [],
                "value_distribution": {}
            }
            
            # Count total and null values
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT({col_name}) as non_null_count,
                    COUNT(DISTINCT {col_name}) as unique_count
                FROM {table_name}
            """)
            
            counts = cursor.fetchone()
            column_analysis["total_count"] = counts[0]
            column_analysis["null_count"] = counts[0] - counts[1]
            column_analysis["unique_count"] = counts[2]
            column_analysis["null_percentage"] = (column_analysis["null_count"] / counts[0]) * 100 if counts[0] > 0 else 0
            
            # Get sample values (non-null)
            cursor.execute(f"""
                SELECT DISTINCT {col_name}
                FROM {table_name}
                WHERE {col_name} IS NOT NULL
                ORDER BY {col_name}
                LIMIT 10
            """)
            
            sample_values = [row[0] for row in cursor.fetchall()]
            column_analysis["sample_values"] = sample_values
            
            # For text columns, get value distribution
            if col_type in ['text', 'character varying', 'varchar']:
                cursor.execute(f"""
                    SELECT {col_name}, COUNT(*) as count
                    FROM {table_name}
                    WHERE {col_name} IS NOT NULL
                    GROUP BY {col_name}
                    ORDER BY count DESC
                    LIMIT 20
                """)
                
                distribution = {}
                for row in cursor.fetchall():
                    distribution[str(row[0])] = row[1]
                column_analysis["value_distribution"] = distribution
            
            # For numeric columns, get statistics
            elif col_type in ['integer', 'bigint', 'numeric', 'real', 'double precision']:
                try:
                    cursor.execute(f"""
                        SELECT 
                            MIN({col_name}) as min_val,
                            MAX({col_name}) as max_val,
                            AVG({col_name}) as avg_val,
                            STDDEV({col_name}) as std_val
                        FROM {table_name}
                        WHERE {col_name} IS NOT NULL
                    """)
                    
                    stats = cursor.fetchone()
                    if stats:
                        column_analysis["statistics"] = {
                            "min": stats[0],
                            "max": stats[1],
                            "average": float(stats[2]) if stats[2] else None,
                            "std_dev": float(stats[3]) if stats[3] else None
                        }
                except Exception as e:
                    logger.warning(f"Could not compute statistics for {col_name}: {e}")
            
            column_details[col_name] = column_analysis
        
        return {
            "table_name": table_name,
            "total_columns": len(column_details),
            "column_details": column_details,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing column details for {table_name}: {e}")
        raise
    finally:
        if should_close:
            conn.close()

def identify_sc_eqtl_relevant_columns(
    table_name: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, List[str]]:
    """
    Identify columns relevant for sc-eQTL analysis based on content analysis.
    
    Args:
        table_name: Name of the table to analyze
        conn: Database connection (optional)
        
    Returns:
        Categorized relevant columns
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        # Get detailed column analysis
        column_details = analyze_column_details(table_name, conn)
        
        # Define sc-eQTL relevant patterns
        relevance_patterns = {
            "species": {
                "keywords": ["organism", "species", "taxon", "scientific_name"],
                "values": ["homo sapiens", "human", "mus musculus", "mouse"]
            },
            "cell_type": {
                "keywords": ["cell_type", "cell_line", "celltype", "cell", "line"],
                "values": ["hela", "293t", "k562", "jurkat", "cell line"]
            },
            "tissue": {
                "keywords": ["tissue", "organ", "anatomical", "body_part", "source"],
                "values": ["brain", "liver", "heart", "lung", "blood", "kidney", "muscle"]
            },
            "sample_info": {
                "keywords": ["sample", "subject", "patient", "donor", "individual"],
                "values": []
            },
            "sequencing": {
                "keywords": ["seq", "rna", "assay", "platform", "instrument", "library", "method"],
                "values": ["10x", "smart-seq", "drop-seq", "illumina", "nextseq", "hiseq"]
            },
            "geographic": {
                "keywords": ["country", "region", "location", "geographic", "nation"],
                "values": ["usa", "china", "uk", "germany", "japan"]
            },
            "age": {
                "keywords": ["age", "years", "old", "birth", "born"],
                "values": []
            },
            "cancer": {
                "keywords": ["cancer", "tumor", "tumour", "malignant", "carcinoma", "adenocarcinoma"],
                "values": ["cancer", "tumor", "normal", "healthy", "control"]
            },
            "publication": {
                "keywords": ["pmid", "doi", "pubmed", "paper", "article", "publication"],
                "values": []
            },
            "database_ids": {
                "keywords": ["geo", "sra", "accession", "id", "gsm", "srx", "srr"],
                "values": []
            }
        }
        
        relevant_columns = {}
        
        for category, patterns in relevance_patterns.items():
            relevant_columns[category] = []
            
            for col_name, col_info in column_details["column_details"].items():
                col_name_lower = col_name.lower()
                
                # Check if column name matches keywords
                name_match = any(keyword in col_name_lower for keyword in patterns["keywords"])
                
                # Check if column values match patterns
                value_match = False
                if patterns["values"]:
                    sample_values = [str(v).lower() for v in col_info.get("sample_values", [])]
                    value_distribution = {k.lower(): v for k, v in col_info.get("value_distribution", {}).items()}
                    
                    value_match = any(
                        any(pattern in value for pattern in patterns["values"])
                        for value in sample_values + list(value_distribution.keys())
                    )
                
                if name_match or value_match:
                    relevant_columns[category].append({
                        "column_name": col_name,
                        "match_type": "name" if name_match else "value",
                        "data_type": col_info["data_type"],
                        "unique_count": col_info["unique_count"],
                        "null_percentage": col_info["null_percentage"],
                        "sample_values": col_info["sample_values"][:5]  # First 5 samples
                    })
        
        return relevant_columns
        
    finally:
        if should_close:
            conn.close()

def generate_table_profile(
    table_name: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive table profile for sc-eQTL analysis.
    
    Args:
        table_name: Name of the table to profile
        conn: Database connection (optional)
        
    Returns:
        Comprehensive table profile
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        # Get basic schema info
        schema_info = analyze_table_schema(table_name, conn)
        
        # Get detailed column analysis
        column_details = analyze_column_details(table_name, conn)
        
        # Identify relevant columns
        relevant_columns = identify_sc_eqtl_relevant_columns(table_name, conn)
        
        # Generate profile
        profile = {
            "table_name": table_name,
            "basic_info": {
                "row_count": schema_info["row_count"],
                "column_count": len(schema_info["columns"]),
                "table_size": schema_info["table_size"]
            },
            "column_analysis": column_details,
            "sc_eqtl_relevance": relevant_columns,
            "data_quality": {
                "columns_with_nulls": sum(1 for col in column_details["column_details"].values() if col["null_count"] > 0),
                "high_null_columns": [
                    col["column_name"] for col in column_details["column_details"].values() 
                    if col["null_percentage"] > 50
                ],
                "low_diversity_columns": [
                    col["column_name"] for col in column_details["column_details"].values() 
                    if col["unique_count"] < 5 and col["total_count"] > 100
                ]
            },
            "profile_timestamp": datetime.now().isoformat()
        }
        
        return profile
        
    finally:
        if should_close:
            conn.close() 