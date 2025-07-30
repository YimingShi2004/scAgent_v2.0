"""
Database query utilities for scAgent.
"""

import psycopg2
import psycopg2.extras
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
from .connect import get_connection, get_cursor

logger = logging.getLogger(__name__)

def execute_query(
    query: str,
    params: Optional[Tuple] = None,
    conn: Optional[psycopg2.extensions.connection] = None,
    fetch_all: bool = True
) -> List[Dict[str, Any]]:
    """
    Execute a SQL query and return results.
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
        conn: Database connection (optional)
        fetch_all: Whether to fetch all results or just one
        
    Returns:
        List of dictionaries containing query results
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        with get_cursor(conn, psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            
            if fetch_all:
                results = cur.fetchall()
            else:
                results = [cur.fetchone()]
            
            return [dict(row) for row in results if row is not None]
            
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        logger.error(f"Query: {query}")
        raise
    finally:
        if should_close:
            conn.close()

def query_geo_master(
    limit: int = 1000,
    offset: int = 0,
    conditions: Optional[Dict[str, Any]] = None,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Query the geo_master table with optional filtering.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        conditions: Dictionary of column-value pairs for filtering
        conn: Database connection (optional)
        
    Returns:
        List of dictionaries containing geo_master records
    """
    query = "SELECT * FROM geo_master"
    params = []
    
    # Add WHERE conditions if provided
    if conditions:
        where_clauses = []
        for column, value in conditions.items():
            if isinstance(value, list):
                # Handle IN clause
                placeholders = ','.join(['%s'] * len(value))
                where_clauses.append(f"{column} IN ({placeholders})")
                params.extend(value)
            elif isinstance(value, dict) and 'like' in value:
                # Handle LIKE clause
                where_clauses.append(f"{column} ILIKE %s")
                params.append(f"%{value['like']}%")
            else:
                # Handle equality
                where_clauses.append(f"{column} = %s")
                params.append(value)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
    
    # Add LIMIT and OFFSET
    query += f" LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    return execute_query(query, tuple(params), conn)

def query_sra_master(
    limit: int = 1000,
    offset: int = 0,
    conditions: Optional[Dict[str, Any]] = None,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Query the sra_master table with optional filtering.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        conditions: Dictionary of column-value pairs for filtering
        conn: Database connection (optional)
        
    Returns:
        List of dictionaries containing sra_master records
    """
    query = "SELECT * FROM sra_master"
    params = []
    
    # Add WHERE conditions if provided
    if conditions:
        where_clauses = []
        for column, value in conditions.items():
            if isinstance(value, list):
                # Handle IN clause
                placeholders = ','.join(['%s'] * len(value))
                where_clauses.append(f"{column} IN ({placeholders})")
                params.extend(value)
            elif isinstance(value, dict) and 'like' in value:
                # Handle LIKE clause
                where_clauses.append(f"{column} ILIKE %s")
                params.append(f"%{value['like']}%")
            else:
                # Handle equality
                where_clauses.append(f"{column} = %s")
                params.append(value)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
    
    # Add LIMIT and OFFSET
    query += f" LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    return execute_query(query, tuple(params), conn)

def find_scrna_datasets(
    limit: int = 1000,
    organisms: Optional[List[str]] = None,
    tissues: Optional[List[str]] = None,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Find single-cell RNA-seq datasets suitable for sc-eQTL analysis.
    
    Args:
        limit: Maximum number of records to return
        organisms: List of organism names to filter by
        tissues: List of tissue names to filter by
        conn: Database connection (optional)
        
    Returns:
        List of dictionaries containing relevant datasets
    """
    # First, find sc-RNA datasets from GEO
    geo_query = """
    SELECT 
        g.*,
        'GEO' as source_type
    FROM geo_master g
    WHERE (
        g.title ILIKE %s OR g.title ILIKE %s OR g.title ILIKE %s OR g.title ILIKE %s OR
        g.summary ILIKE %s OR g.summary ILIKE %s OR g.summary ILIKE %s OR g.summary ILIKE %s
    )
    AND g.status = 'Public'
    """
    
    params = []
    scrna_terms = ["%single cell%", "%single-cell%", "%scRNA%", "%sc-RNA%"]
    
    # Add terms for both title and summary searches
    params.extend(scrna_terms)  # for title
    params.extend(scrna_terms)  # for summary
    
    # Add organism filter
    if organisms:
        geo_query += " AND g.organism IN ({})".format(','.join(['%s'] * len(organisms)))
        params.extend(organisms)
    
    # Add tissue filter (search in title and summary)
    if tissues:
        tissue_conditions = []
        for tissue in tissues:
            tissue_conditions.extend([
                "g.title ILIKE %s",
                "g.summary ILIKE %s"
            ])
            params.extend([f"%{tissue}%", f"%{tissue}%"])
        
        if tissue_conditions:
            geo_query += " AND (" + " OR ".join(tissue_conditions) + ")"
    
    # Order by relevance (more recent first)
    geo_query += " ORDER BY g.submission_date DESC"
    
    # Add limit
    geo_query += f" LIMIT %s"
    params.append(limit)
    
    # Execute GEO query
    geo_results = execute_query(geo_query, tuple(params), conn)
    
    # Then find corresponding SRA data
    if geo_results:
        # Get SRA data for these datasets
        sra_query = """
        SELECT 
            s.*,
            'SRA' as source_type
        FROM sra_master s
        WHERE (
            s.study_title ILIKE %s OR s.study_title ILIKE %s OR s.study_title ILIKE %s OR s.study_title ILIKE %s OR
            s.study_abstract ILIKE %s OR s.study_abstract ILIKE %s OR s.study_abstract ILIKE %s OR s.study_abstract ILIKE %s
        )
        AND s.library_strategy = 'RNA-Seq'
        AND s.library_source = 'TRANSCRIPTOMIC'
        """
        
        sra_params = scrna_terms + scrna_terms  # for title and abstract
        
        # Add organism filter for SRA
        if organisms:
            sra_query += " AND s.organism IN ({})".format(','.join(['%s'] * len(organisms)))
            sra_params.extend(organisms)
        
        # Add tissue filter for SRA
        if tissues:
            sra_query += " AND s.tissue IN ({})".format(','.join(['%s'] * len(tissues)))
            sra_params.extend(tissues)
        
        sra_query += f" LIMIT %s"
        sra_params.append(limit)
        
        sra_results = execute_query(sra_query, tuple(sra_params), conn)
        
        # Combine results
        combined_results = []
        for geo_record in geo_results:
            # Add GEO record
            combined_results.append(geo_record)
            
        for sra_record in sra_results:
            # Add SRA record
            combined_results.append(sra_record)
            
        return combined_results
    
    return geo_results

def get_dataset_statistics(
    table_name: str,
    conn: Optional[psycopg2.extensions.connection] = None
) -> Dict[str, Any]:
    """
    Get statistical overview of a dataset table.
    
    Args:
        table_name: Name of the table (geo_master or sra_master)
        conn: Database connection (optional)
        
    Returns:
        Dictionary containing dataset statistics
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        stats = {}
        
        with get_cursor(conn, psycopg2.extras.RealDictCursor) as cur:
            # Total records
            cur.execute(f"SELECT COUNT(*) as total_records FROM {table_name}")
            stats['total_records'] = cur.fetchone()['total_records']
            
            if table_name == 'geo_master':
                # Organism distribution
                cur.execute("""
                    SELECT organism, COUNT(*) as count
                    FROM geo_master
                    WHERE organism IS NOT NULL
                    GROUP BY organism
                    ORDER BY count DESC
                    LIMIT 10
                """)
                stats['top_organisms'] = [dict(row) for row in cur.fetchall()]
                
                # Status distribution
                cur.execute("""
                    SELECT status, COUNT(*) as count
                    FROM geo_master
                    GROUP BY status
                """)
                stats['status_distribution'] = [dict(row) for row in cur.fetchall()]
                
                # Recent submissions (last 5 years)
                cur.execute("""
                    SELECT 
                        DATE_PART('year', submission_date) as year,
                        COUNT(*) as count
                    FROM geo_master
                    WHERE submission_date >= NOW() - INTERVAL '5 years'
                    GROUP BY year
                    ORDER BY year DESC
                """)
                stats['recent_submissions'] = [dict(row) for row in cur.fetchall()]
                
            elif table_name == 'sra_master':
                # Platform distribution
                cur.execute("""
                    SELECT platform, COUNT(*) as count
                    FROM sra_master
                    WHERE platform IS NOT NULL
                    GROUP BY platform
                    ORDER BY count DESC
                    LIMIT 10
                """)
                stats['top_platforms'] = [dict(row) for row in cur.fetchall()]
                
                # Library strategy distribution
                cur.execute("""
                    SELECT library_strategy, COUNT(*) as count
                    FROM sra_master
                    WHERE library_strategy IS NOT NULL
                    GROUP BY library_strategy
                    ORDER BY count DESC
                """)
                stats['library_strategies'] = [dict(row) for row in cur.fetchall()]
                
                # Library source distribution
                cur.execute("""
                    SELECT library_source, COUNT(*) as count
                    FROM sra_master
                    WHERE library_source IS NOT NULL
                    GROUP BY library_source
                    ORDER BY count DESC
                """)
                stats['library_sources'] = [dict(row) for row in cur.fetchall()]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting dataset statistics for {table_name}: {e}")
        raise
    finally:
        if should_close:
            conn.close()

def search_datasets_by_keywords(
    keywords: List[str],
    table_name: str = 'geo_master',
    search_columns: Optional[List[str]] = None,
    limit: int = 100,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Search datasets by keywords in specified columns.
    
    Args:
        keywords: List of keywords to search for
        table_name: Table to search in (geo_master or sra_master)
        search_columns: Columns to search in (defaults to text columns)
        limit: Maximum number of results
        conn: Database connection (optional)
        
    Returns:
        List of matching records
    """
    if search_columns is None:
        if table_name == 'geo_master':
            search_columns = ['title', 'summary', 'organism']
        else:
            search_columns = ['study_title', 'study_abstract', 'library_strategy']
    
    # Build search query
    search_conditions = []
    params = []
    
    for keyword in keywords:
        keyword_conditions = []
        for column in search_columns:
            keyword_conditions.append(f"{column} ILIKE %s")
            params.append(f"%{keyword}%")
        search_conditions.append("(" + " OR ".join(keyword_conditions) + ")")
    
    query = f"""
    SELECT * FROM {table_name}
    WHERE {' AND '.join(search_conditions)}
    ORDER BY submission_date DESC
    LIMIT %s
    """
    params.append(limit)
    
    return execute_query(query, tuple(params), conn)

def export_query_results(
    results: List[Dict[str, Any]],
    output_file: str,
    format: str = 'csv'
) -> str:
    """
    Export query results to a file.
    
    Args:
        results: Query results to export
        output_file: Output file path
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        Path to the exported file
    """
    if not results:
        logger.warning("No results to export")
        return output_file
    
    df = pd.DataFrame(results)
    
    if format.lower() == 'csv':
        df.to_csv(output_file, index=False)
    elif format.lower() == 'json':
        df.to_json(output_file, orient='records', indent=2)
    elif format.lower() == 'excel':
        df.to_excel(output_file, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Exported {len(results)} records to {output_file}")
    return output_file

def get_geo_datasets(
    limit: int = 1000,
    organisms: Optional[List[str]] = None,
    tissues: Optional[List[str]] = None,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Get GEO datasets with optional filtering.
    
    Args:
        limit: Maximum number of records to return
        organisms: List of organism names to filter by
        tissues: List of tissue names to filter by
        conn: Database connection (optional)
        
    Returns:
        List of GEO dataset records
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        query = """
        SELECT * FROM geo_master
        WHERE status = 'Public'
        """
        params = []
        
        # Add organism filter
        if organisms:
            query += " AND organism IN ({})".format(','.join(['%s'] * len(organisms)))
            params.extend(organisms)
        
        # Add tissue filter (search in title and summary)
        if tissues:
            tissue_conditions = []
            for tissue in tissues:
                tissue_conditions.extend([
                    "title ILIKE %s",
                    "summary ILIKE %s"
                ])
                params.extend([f"%{tissue}%", f"%{tissue}%"])
            
            if tissue_conditions:
                query += " AND (" + " OR ".join(tissue_conditions) + ")"
        
        # Order by submission date
        query += " ORDER BY submission_date DESC"
        
        # Add limit
        query += f" LIMIT %s"
        params.append(limit)
        
        return execute_query(query, params, conn)
        
    finally:
        if should_close:
            conn.close()


def get_sra_datasets(
    limit: int = 1000,
    organisms: Optional[List[str]] = None,
    tissues: Optional[List[str]] = None,
    conn: Optional[psycopg2.extensions.connection] = None
) -> List[Dict[str, Any]]:
    """
    Get SRA datasets with optional filtering.
    
    Args:
        limit: Maximum number of records to return
        organisms: List of organism names to filter by
        tissues: List of tissue names to filter by
        conn: Database connection (optional)
        
    Returns:
        List of SRA dataset records
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    
    try:
        query = """
        SELECT * FROM sra_master
        WHERE 1=1
        """
        params = []
        
        # Add organism filter
        if organisms:
            query += " AND organism IN ({})".format(','.join(['%s'] * len(organisms)))
            params.extend(organisms)
        
        # Add tissue filter (search in study_title and study_abstract)
        if tissues:
            tissue_conditions = []
            for tissue in tissues:
                tissue_conditions.extend([
                    "study_title ILIKE %s",
                    "study_abstract ILIKE %s",
                    "tissue ILIKE %s"
                ])
                params.extend([f"%{tissue}%", f"%{tissue}%", f"%{tissue}%"])
            
            if tissue_conditions:
                query += " AND (" + " OR ".join(tissue_conditions) + ")"
        
        # Order by id (since submission_date doesn't exist)
        query += " ORDER BY id DESC"
        
        # Add limit
        query += f" LIMIT %s"
        params.append(limit)
        
        return execute_query(query, params, conn)
        
    finally:
        if should_close:
            conn.close() 