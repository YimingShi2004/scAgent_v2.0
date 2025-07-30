"""
Database connection and operations for scAgent.
"""

from .connect import get_connection, test_connection
from .schema import analyze_table_schema, get_table_info, export_schema_report
from .query import (
    query_geo_master, 
    query_sra_master, 
    execute_query, 
    find_scrna_datasets,
    export_query_results,
    get_dataset_statistics
)

__all__ = [
    "get_connection",
    "test_connection", 
    "analyze_table_schema",
    "get_table_info",
    "export_schema_report",
    "query_geo_master",
    "query_sra_master",
    "execute_query",
    "find_scrna_datasets",
    "export_query_results",
    "get_dataset_statistics"
] 