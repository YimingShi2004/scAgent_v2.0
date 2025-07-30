"""
Main CLI interface for scAgent.
"""

import argparse
import sys
import logging
from typing import Optional, List
import io

from ..db import test_connection as db_test_connection
from ..models import get_qwen_client
from .commands import (
    test_connection,
    analyze_schema,
    analyze_geo,
    analyze_sra,
    find_eqtl_data,
    clean_data,
    batch_assess,
    generate_downloads,
    comprehensive_report,
    show_help,
    deep_analyze,
    full_export,
    comprehensive_clean
)

# Import enhanced commands
from .enhanced_commands import add_enhanced_commands

def force_utf8_stdout_stderr():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

force_utf8_stdout_stderr()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    
    parser = argparse.ArgumentParser(
        description="scAgent: AI agent for identifying and cleaning sc-RNA data suitable for sc-eQTL analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        title="Commands",
        description="Available scAgent commands",
        dest="command",
        help="Command to run"
    )
    
    # Test connection command
    test_parser = subparsers.add_parser(
        "test-connection",
        help="Test database and model API connections"
    )
    test_parser.add_argument(
        "--db-only",
        action="store_true",
        help="Test only database connection"
    )
    test_parser.add_argument(
        "--model-only",
        action="store_true",
        help="Test only model API connection"
    )
    
    # Analyze schema command
    schema_parser = subparsers.add_parser(
        "analyze-schema",
        help="Analyze database table schemas"
    )
    schema_parser.add_argument(
        "--tables",
        nargs="+",
        default=["geo_master", "sra_master"],
        help="Tables to analyze (default: geo_master sra_master)"
    )
    schema_parser.add_argument(
        "--output",
        help="Output file for schema report"
    )
    
    # Analyze GEO command
    geo_parser = subparsers.add_parser(
        "analyze-geo",
        help="Analyze geo_master table data"
    )
    geo_parser.add_argument(
        "--max-records",
        type=int,
        default=1000,
        help="Maximum records to analyze (default: 1000)"
    )
    geo_parser.add_argument(
        "--organisms",
        nargs="+",
        help="Filter by organism names"
    )
    geo_parser.add_argument(
        "--output",
        help="Output file for results"
    )
    
    # Analyze SRA command
    sra_parser = subparsers.add_parser(
        "analyze-sra",
        help="Analyze sra_master table data"
    )
    sra_parser.add_argument(
        "--max-records",
        type=int,
        default=1000,
        help="Maximum records to analyze (default: 1000)"
    )
    sra_parser.add_argument(
        "--platforms",
        nargs="+",
        help="Filter by platform names"
    )
    sra_parser.add_argument(
        "--output",
        help="Output file for results"
    )
    
    # Find eQTL data command
    eqtl_parser = subparsers.add_parser(
        "find-eqtl-data",
        help="Find datasets suitable for sc-eQTL analysis"
    )
    eqtl_parser.add_argument(
        "--max-datasets",
        type=int,
        default=100,
        help="Maximum datasets to find (default: 100)"
    )
    eqtl_parser.add_argument(
        "--organisms",
        nargs="+",
        help="Filter by organism names"
    )
    eqtl_parser.add_argument(
        "--tissues",
        nargs="+",
        help="Filter by tissue names"
    )
    eqtl_parser.add_argument(
        "--output",
        help="Output file for results"
    )
    eqtl_parser.add_argument(
        "--format",
        choices=["csv", "json", "excel"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    # Clean data command
    clean_parser = subparsers.add_parser(
        'clean-data',
        help='Clean and filter table data for sc-eQTL analysis'
    )
    clean_parser.add_argument('--table', required=True, choices=['geo_master', 'sra_master'], help='Table to clean')
    clean_parser.add_argument('--output', required=True, help='Output file path')
    clean_parser.add_argument('--format', choices=['csv', 'json', 'tsv'], default='csv', help='Output format')
    clean_parser.add_argument('--max-records', type=int, default=1000, help='Maximum records to process')
    clean_parser.set_defaults(func=clean_data)
    
    # Batch assessment command
    batch_parser = subparsers.add_parser(
        'batch-assess',
        help='Perform batch quality assessment on datasets'
    )
    batch_parser.add_argument('--source', choices=['geo', 'sra', 'both'], default='both', help='Data source to assess')
    batch_parser.add_argument('--organisms', nargs='+', help='Filter by organism names')
    batch_parser.add_argument('--tissues', nargs='+', help='Filter by tissue names')
    batch_parser.add_argument('--max-records', type=int, default=1000, help='Maximum records to assess')
    batch_parser.add_argument('--output', help='Output file for detailed assessment (JSON)')
    batch_parser.set_defaults(func=batch_assess)
    
    # Generate downloads command
    download_parser = subparsers.add_parser(
        'generate-downloads',
        help='Generate SRA download script for selected datasets'
    )
    download_parser.add_argument('--organisms', nargs='+', help='Filter by organism names')
    download_parser.add_argument('--tissues', nargs='+', help='Filter by tissue names')
    download_parser.add_argument('--max-records', type=int, default=100, help='Maximum datasets to include')
    download_parser.add_argument('--quality-filter', action='store_true', help='Only include high-quality datasets')
    download_parser.add_argument('--output-dir', default='./sra_downloads', help='Output directory for downloads')
    download_parser.add_argument('--script-name', default='download_sra.sh', help='Name of the download script')
    download_parser.set_defaults(func=generate_downloads)
    
    # Comprehensive report command
    report_parser = subparsers.add_parser(
        'comprehensive-report',
        help='Generate comprehensive data quality report'
    )
    report_parser.add_argument('--organisms', nargs='+', help='Filter by organism names')
    report_parser.add_argument('--tissues', nargs='+', help='Filter by tissue names')
    report_parser.add_argument('--max-records', type=int, default=1000, help='Maximum datasets to analyze')
    report_parser.add_argument('--include-ai', action='store_true', help='Include AI analysis in report')
    report_parser.add_argument('--output', required=True, help='Output file for comprehensive report (JSON)')
    report_parser.set_defaults(func=comprehensive_report)
    
    # Help command
    help_parser = subparsers.add_parser(
        'help',
        help='Show detailed help and usage examples'
    )
    help_parser.set_defaults(func=show_help)
    
    # Deep analyze command
    deep_parser = subparsers.add_parser(
        'deep-analyze',
        help='Perform deep analysis of table structure and content'
    )
    deep_parser.add_argument('--table', required=True, choices=['geo_master', 'sra_master'], help='Table to analyze')
    deep_parser.add_argument('--show-details', action='store_true', help='Show detailed column analysis')
    deep_parser.add_argument('--output', help='Output file for detailed analysis (JSON)')
    deep_parser.set_defaults(func=deep_analyze)
    
    # Full export command
    export_parser = subparsers.add_parser(
        'full-export',
        help='Export complete table data for server-side processing'
    )
    export_parser.add_argument('--table', required=True, choices=['geo_master', 'sra_master'], help='Table to export')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['csv', 'json', 'parquet'], default='csv', help='Export format')
    export_parser.add_argument('--show-sample', action='store_true', help='Show sample data preview')
    export_parser.add_argument('--include-metadata', action='store_true', help='Generate metadata file')
    export_parser.set_defaults(func=full_export)
    
    # Comprehensive clean command
    clean_parser = subparsers.add_parser(
        'comprehensive-clean',
        help='Comprehensive data cleaning and filtering for sc-eQTL analysis'
    )
    clean_parser.add_argument('--output', required=True, help='Output file for cleaned data')
    clean_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Output format')
    clean_parser.add_argument('--include-geo', action='store_true', default=True, help='Include GEO data')
    clean_parser.add_argument('--include-sra', action='store_true', default=True, help='Include SRA data')
    clean_parser.add_argument('--species', nargs='+', help='Required species (default: Homo sapiens)')
    clean_parser.add_argument('--include-cell-lines', action='store_true', help='Include cell line data')
    clean_parser.add_argument('--require-publication', action='store_true', help='Require publication info')
    clean_parser.add_argument('--require-sample-size', action='store_true', help='Require sample size info')
    clean_parser.add_argument('--require-country', action='store_true', help='Require country info')
    clean_parser.add_argument('--require-age', action='store_true', help='Require age info')
    clean_parser.add_argument('--require-tumor-status', action='store_true', help='Require tumor status annotation')
    clean_parser.add_argument('--include-report', action='store_true', help='Generate filter report')
    clean_parser.add_argument('--use-ai', action='store_true', default=True, help='Use AI for intelligent filtering')
    clean_parser.add_argument('--no-ai', action='store_true', help='Disable AI filtering')
    clean_parser.add_argument('--ai-batch-size', type=int, default=5, help='Batch size for AI processing')
    clean_parser.add_argument('--limit', type=int, help='Limit number of records to process (for testing)')
    clean_parser.set_defaults(func=comprehensive_clean)
    
    # Add enhanced commands for merged table processing
    add_enhanced_commands(subparsers)
    
    return parser

def main() -> int:
    """Main entry point for the CLI."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Route to appropriate command handler
        if args.command == "test-connection":
            return test_connection(args)
        elif args.command == "analyze-schema":
            return analyze_schema(args)
        elif args.command == "analyze-geo":
            return analyze_geo(args)
        elif args.command == "analyze-sra":
            return analyze_sra(args)
        elif args.command == "find-eqtl-data":
            return find_eqtl_data(args)
        elif args.command == "clean-data":
            return clean_data(args)
        elif args.command == "batch-assess":
            return batch_assess(args)
        elif args.command == "generate-downloads":
            return generate_downloads(args)
        elif args.command == "comprehensive-report":
            return comprehensive_report(args)
        elif args.command == "help":
            return show_help(args)
        elif args.command == "deep-analyze":
            return deep_analyze(args)
        elif args.command == "full-export":
            return full_export(args)
        elif args.command == "comprehensive-clean":
            return comprehensive_clean(args)
        # Enhanced commands - use function stored in args by enhanced_commands
        elif hasattr(args, 'func'):
            args.func(args)
            return 0
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 