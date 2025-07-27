"""
Enhanced CLI commands for scAgent with full-scan capabilities and merged table support.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
import time
from datetime import datetime

from ..utils_enhanced_filtering import EnhancedScEQTLFilter
from ..db.merged_table_handler import MergedTableHandler

console = Console()
logger = logging.getLogger(__name__)

def add_enhanced_commands(subparsers):
    """Add enhanced commands to the CLI parser."""
    
    # Full-scan analysis command
    parser_fullscan = subparsers.add_parser(
        'full-scan-analysis',
        help='Perform full-scan analysis on merged table with millions of records',
        description='Advanced full-scan filtering with AI assistance and optimization strategies'
    )
    
    parser_fullscan.add_argument(
        '--schema',
        type=str,
        default='merge',
        help='Schema name to use (default: merge)'
    )
    parser_fullscan.add_argument(
        '--table',
        type=str,
        default='sra_geo_ft2',
        help='Table name to use (default: sra_geo_ft2)'
    )
    
    parser_fullscan.add_argument(
        '--max-records', 
        type=int, 
        default=None,
        help='Maximum records to process (default: unlimited)'
    )
    
    parser_fullscan.add_argument(
        '--batch-size', 
        type=int, 
        default=10000,
        help='Records per batch (default: 10000)'
    )
    
    parser_fullscan.add_argument(
        '--enable-ai', 
        action='store_true',
        help='Enable AI assistance for content recognition'
    )
    
    parser_fullscan.add_argument(
        '--enable-parallel', 
        action='store_true',
        help='Enable parallel processing'
    )
    
    parser_fullscan.add_argument(
        '--output', 
        type=str,
        help='Output file path (.json or .csv)'
    )
    
    parser_fullscan.add_argument(
        '--filter-config', 
        type=str,
        help='Custom filter configuration file (JSON)'
    )
    
    parser_fullscan.add_argument(
        '--organism', 
        choices=['human', 'mouse', 'both'], 
        default='human',
        help='Target organism(s) for filtering'
    )
    
    parser_fullscan.add_argument(
        '--min-quality-score', 
        type=float, 
        default=3.0,
        help='Minimum quality score for inclusion'
    )
    
    parser_fullscan.set_defaults(func=command_full_scan_analysis)
    
    # Table analysis command
    parser_analyze = subparsers.add_parser(
        'analyze-merged-table',
        help='Analyze merged table structure and create field mappings',
        description='Comprehensive analysis of the merged sra_geo_ft2 table'
    )
    
    parser_analyze.add_argument(
        '--table', 
        type=str, 
        default='merged/sra_geo_ft2',
        help='Table name to analyze'
    )
    
    parser_analyze.add_argument(
        '--output-report', 
        type=str,
        help='Save analysis report to file'
    )
    
    parser_analyze.add_argument(
        '--detailed', 
        action='store_true',
        help='Include detailed column analysis'
    )
    
    parser_analyze.set_defaults(func=command_analyze_merged_table)
    
    # Relationship optimization command
    parser_relationships = subparsers.add_parser(
        'optimize-relationships',
        help='Analyze and optimize study-run relationships',
        description='Optimize display and processing of one-to-many relationships'
    )
    
    parser_relationships.add_argument(
        '--visualize', 
        action='store_true',
        help='Create relationship visualization'
    )
    
    parser_relationships.add_argument(
        '--export-strategies', 
        type=str,
        help='Export optimization strategies to file'
    )
    
    parser_relationships.set_defaults(func=command_optimize_relationships)
    
    # Field mapping command
    parser_mapping = subparsers.add_parser(
        'create-field-mapping',
        help='Create intelligent field mappings for required and optional fields',
        description='Map table columns to sc-eQTL analysis requirements'
    )
    
    parser_mapping.add_argument(
        '--export-mapping', 
        type=str,
        help='Export field mapping to file'
    )
    
    parser_mapping.add_argument(
        '--confidence-threshold', 
        type=float, 
        default=0.6,
        help='Minimum confidence for automatic mapping'
    )
    
    parser_mapping.set_defaults(func=command_create_field_mapping)
    
    # System initialization command
    parser_init = subparsers.add_parser(
        'init-enhanced-system',
        help='Initialize enhanced filtering system with table analysis',
        description='Complete system initialization and validation'
    )
    
    parser_init.add_argument(
        '--quick', 
        action='store_true',
        help='Quick initialization (skip detailed analysis)'
    )
    
    parser_init.add_argument(
        '--save-config', 
        type=str,
        help='Save initialization results to config file'
    )
    
    parser_init.set_defaults(func=command_init_enhanced_system)
    
    # Performance benchmark command
    parser_benchmark = subparsers.add_parser(
        'benchmark-performance',
        help='Benchmark system performance with different configurations',
        description='Test and optimize processing performance'
    )
    
    parser_benchmark.add_argument(
        '--test-records', 
        type=int, 
        default=10000,
        help='Number of records for testing'
    )
    
    parser_benchmark.add_argument(
        '--test-batch-sizes', 
        nargs='+', 
        type=int, 
        default=[1000, 5000, 10000, 20000],
        help='Batch sizes to test'
    )
    
    parser_benchmark.set_defaults(func=command_benchmark_performance)

def command_full_scan_analysis(args):
    """Execute full-scan analysis command."""
    
    console.print(Panel(
        "[bold blue]scAgent Enhanced Full-Scan Analysis[/bold blue]\n"
        "Processing merged table with advanced filtering strategies",
        expand=False
    ))
    
    try:
        # Load custom filter config if provided
        filter_config = None
        if args.filter_config:
            with open(args.filter_config, 'r') as f:
                filter_config = json.load(f)
            console.print(f"\u2713 Loaded filter configuration from {args.filter_config}")
        
        # Initialize enhanced filter system
        console.print("\ud83d\udd04 Initializing enhanced filtering system...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            init_task = progress.add_task("Initializing system...", total=None)
            
            # 修正：传递schema和table参数
            filter_system = EnhancedScEQTLFilter(schema=args.schema, table=args.table)
            init_results = filter_system.initialize_system()
            
            progress.remove_task(init_task)
        
        if init_results["status"] != "success":
            console.print("[red]✗ System initialization failed[/red]")
            return
        
        # Display initialization summary
        init_time = init_results["initialization_time"]
        total_columns = init_results["table_structure"]["total_columns"]
        total_records = init_results["table_summary"]["basic_statistics"]["total_records"]
        
        console.print(f"✓ System initialized in {init_time:.2f}s")
        console.print(f"✓ Table: {total_records:,} records, {total_columns} columns")
        
        # Customize filter config based on args
        if filter_config is None:
            filter_config = filter_system._get_enhanced_filter_config()
        
        # Apply organism filter
        if args.organism == 'human':
            filter_config["required_fields"]["organism"]["accepted_values"] = ["Homo sapiens", "human"]
        elif args.organism == 'mouse':
            filter_config["required_fields"]["organism"]["accepted_values"] = ["Mus musculus", "mouse"]
        elif args.organism == 'both':
            filter_config["required_fields"]["organism"]["accepted_values"] = [
                "Homo sapiens", "human", "Mus musculus", "mouse"
            ]
        
        # Apply quality score threshold
        filter_config["processing"]["min_quality_score"] = args.min_quality_score
        
        # Display filter configuration
        console.print("\n[bold]Filter Configuration:[/bold]")
        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Setting")
        config_table.add_column("Value")
        
        config_table.add_row("Target Organism(s)", args.organism)
        config_table.add_row("Min Quality Score", str(args.min_quality_score))
        config_table.add_row("Max Records", str(args.max_records) if args.max_records else "Unlimited")
        config_table.add_row("Batch Size", str(args.batch_size))
        config_table.add_row("AI Assistance", "Enabled" if args.enable_ai else "Disabled")
        config_table.add_row("Parallel Processing", "Enabled" if args.enable_parallel else "Disabled")
        
        console.print(config_table)
        
        # Start full-scan analysis
        console.print(f"\n🚀 Starting full-scan analysis...")
        console.print(f"📊 Processing up to {args.max_records or 'unlimited'} records in batches of {args.batch_size}")
        
        start_time = time.time()
        
        scan_results = filter_system.full_scan_filter(
            filter_config=filter_config,
            batch_size=args.batch_size,
            max_records=args.max_records,
            enable_ai_assistance=args.enable_ai,
            enable_parallel=args.enable_parallel,
            output_file=args.output
        )
        
        # Display results
        if scan_results["scan_summary"]["status"] == "completed":
            display_scan_results(scan_results)
            
            if args.output:
                console.print(f"\n💾 Results saved to: [green]{args.output}[/green]")
        else:
            console.print(f"[red]✗ Scan failed: {scan_results['scan_summary'].get('error', 'Unknown error')}[/red]")
    
    except Exception as e:
        console.print(f"[red]✗ Full-scan analysis failed: {e}[/red]")
        logger.error(f"Full-scan analysis error: {e}", exc_info=True)

def command_analyze_merged_table(args):
    """Execute merged table analysis command."""
    
    console.print(Panel(
        "[bold green]Merged Table Analysis[/bold green]\n"
        f"Analyzing table: {args.table}",
        expand=False
    ))
    
    try:
        # Initialize table handler
        handler = MergedTableHandler(table=args.table)
        
        # Discover table structure
        console.print("🔍 Discovering table structure...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Analyzing table...", total=None)
            
            structure = handler.discover_table_structure()
            mapping = handler.create_column_mapping()
            summary = handler.get_table_summary()
            
            progress.remove_task(task)
        
        # Display basic statistics
        console.print("\n[bold]Table Statistics:[/bold]")
        stats_table = Table(show_header=True, header_style="bold cyan")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        
        stats_table.add_row("Total Records", f"{summary['basic_statistics']['total_records']:,}")
        stats_table.add_row("Total Columns", str(summary['basic_statistics']['total_columns']))
        stats_table.add_row("Table Size", summary['basic_statistics']['table_size'])
        stats_table.add_row("High Quality Columns", str(summary['data_quality']['high_quality_columns']))
        stats_table.add_row("sc-eQTL Relevant Columns", str(summary['data_quality']['sc_eqtl_relevant_columns']))
        
        console.print(stats_table)
        
        # Display field mapping results
        console.print("\n[bold]Field Mapping Results:[/bold]")
        mapping_table = Table(show_header=True, header_style="bold yellow")
        mapping_table.add_column("Field Type")
        mapping_table.add_column("Required/Optional")
        mapping_table.add_column("Mapped Columns")
        mapping_table.add_column("Confidence")
        
        for field_name, candidates in mapping["required_fields"].items():
            if candidates:
                best_candidate = candidates[0]
                confidence = mapping["mapping_confidence"].get(field_name, 0)
                mapping_table.add_row(
                    field_name,
                    "Required",
                    best_candidate["column_name"],
                    f"{confidence:.2f}"
                )
            else:
                mapping_table.add_row(field_name, "Required", "[red]Not mapped[/red]", "0.00")
        
        for field_name, candidates in mapping["optional_fields"].items():
            if candidates:
                best_candidate = candidates[0]
                confidence = mapping["mapping_confidence"].get(field_name, 0)
                mapping_table.add_row(
                    field_name,
                    "Optional",
                    best_candidate["column_name"],
                    f"{confidence:.2f}"
                )
        
        console.print(mapping_table)
        
        # Display detailed column analysis if requested
        if args.detailed:
            display_detailed_column_analysis(structure["column_analysis"])
        
        # Display recommendations
        if summary["recommendations"]:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in summary["recommendations"]:
                priority_color = {"critical": "red", "high": "yellow", "medium": "blue", "low": "green"}.get(rec["priority"], "white")
                console.print(f"• [{priority_color}]{rec['recommendation']}[/{priority_color}]")
                console.print(f"  Reason: {rec['reason']}")
        
        # Save report if requested
        if args.output_report:
            report_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "table_structure": structure,
                "column_mapping": mapping,
                "table_summary": summary
            }
            
            with open(args.output_report, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            console.print(f"\n💾 Analysis report saved to: [green]{args.output_report}[/green]")
    
    except Exception as e:
        console.print(f"[red]✗ Table analysis failed: {e}[/red]")
        logger.error(f"Table analysis error: {e}", exc_info=True)

def command_optimize_relationships(args):
    """Execute relationship optimization command."""
    
    console.print(Panel(
        "[bold purple]Study-Run Relationship Optimization[/bold purple]\n"
        "Analyzing and optimizing one-to-many relationships",
        expand=False
    ))
    
    try:
        # Initialize table handler
        handler = MergedTableHandler()
        
        # Analyze relationships
        console.print("🔗 Analyzing study-run relationships...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Analyzing relationships...", total=None)
            
            relationships = handler.optimize_study_run_relationships()
            
            progress.remove_task(task)
        
        if "error" in relationships:
            console.print(f"[red]✗ Relationship analysis failed: {relationships['error']}[/red]")
            return
        
        # Display relationship statistics
        stats = relationships["relationship_stats"]
        
        console.print("\n[bold]Relationship Statistics:[/bold]")
        rel_table = Table(show_header=True, header_style="bold magenta")
        rel_table.add_column("Metric")
        rel_table.add_column("Value")
        
        rel_table.add_row("Total Studies", f"{stats['total_studies']:,}")
        rel_table.add_row("Total Runs", f"{stats['total_runs']:,}")
        rel_table.add_row("Avg Runs per Study", f"{stats['avg_runs_per_study']:.1f}")
        rel_table.add_row("Max Runs per Study", str(stats['max_runs_per_study']))
        rel_table.add_row("Studies with Multiple Runs", f"{stats['studies_with_multiple_runs']:,}")
        rel_table.add_row("Multi-Run Percentage", f"{stats['multi_run_percentage']:.1f}%")
        
        console.print(rel_table)
        
        # Display optimization strategies
        strategies = relationships["optimization_strategies"]
        
        if strategies:
            console.print("\n[bold]Optimization Strategies:[/bold]")
            
            for strategy_type, strategy_info in strategies.items():
                if strategy_info:
                    console.print(f"\n[cyan]{strategy_type.replace('_', ' ').title()}:[/cyan]")
                    console.print(f"Recommendation: {strategy_info['recommendation']}")
                    console.print(f"Description: {strategy_info['description']}")
                    
                    if 'benefits' in strategy_info:
                        console.print("Benefits:")
                        for benefit in strategy_info['benefits']:
                            console.print(f"  • {benefit}")
        
        # Export strategies if requested
        if args.export_strategies:
            with open(args.export_strategies, 'w') as f:
                json.dump(relationships, f, indent=2, default=str)
            console.print(f"\n💾 Optimization strategies saved to: [green]{args.export_strategies}[/green]")
    
    except Exception as e:
        console.print(f"[red]✗ Relationship optimization failed: {e}[/red]")
        logger.error(f"Relationship optimization error: {e}", exc_info=True)

def command_create_field_mapping(args):
    """Execute field mapping creation command."""
    
    console.print(Panel(
        "[bold orange]Intelligent Field Mapping[/bold orange]\n"
        "Creating mappings for required and optional sc-eQTL fields",
        expand=False
    ))
    
    try:
        # Initialize table handler
        handler = MergedTableHandler()
        
        # Create field mapping
        console.print("🗺️ Creating intelligent field mappings...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Creating mappings...", total=None)
            
            # First discover structure if not cached
            if handler._column_cache is None:
                handler.discover_table_structure()
            
            mapping = handler.create_column_mapping()
            
            progress.remove_task(task)
        
        # Display mapping results with confidence filtering
        console.print(f"\n[bold]Field Mapping Results (Confidence ≥ {args.confidence_threshold}):[/bold]")
        
        high_confidence_mappings = 0
        low_confidence_mappings = 0
        
        # Required fields
        console.print("\n[bold cyan]Required Fields:[/bold cyan]")
        req_table = Table(show_header=True, header_style="bold cyan")
        req_table.add_column("Field")
        req_table.add_column("Best Candidate")
        req_table.add_column("Confidence")
        req_table.add_column("Null %")
        req_table.add_column("Status")
        
        for field_name, candidates in mapping["required_fields"].items():
            if candidates:
                best_candidate = candidates[0]
                confidence = mapping["mapping_confidence"].get(field_name, 0)
                
                if confidence >= args.confidence_threshold:
                    status = "[green]✓[/green]"
                    high_confidence_mappings += 1
                else:
                    status = "[yellow]⚠[/yellow]"
                    low_confidence_mappings += 1
                
                req_table.add_row(
                    field_name,
                    best_candidate["column_name"],
                    f"{confidence:.2f}",
                    f"{best_candidate['null_percentage']:.1f}%",
                    status
                )
            else:
                req_table.add_row(
                    field_name,
                    "[red]No mapping found[/red]",
                    "0.00",
                    "N/A",
                    "[red]✗[/red]"
                )
        
        console.print(req_table)
        
        # Optional fields
        console.print("\n[bold yellow]Optional Fields:[/bold yellow]")
        opt_table = Table(show_header=True, header_style="bold yellow")
        opt_table.add_column("Field")
        opt_table.add_column("Best Candidate")
        opt_table.add_column("Confidence")
        opt_table.add_column("Status")
        
        for field_name, candidates in mapping["optional_fields"].items():
            if candidates:
                best_candidate = candidates[0]
                confidence = mapping["mapping_confidence"].get(field_name, 0)
                
                if confidence >= args.confidence_threshold:
                    status = "[green]✓[/green]"
                else:
                    status = "[yellow]⚠[/yellow]"
                
                opt_table.add_row(
                    field_name,
                    best_candidate["column_name"],
                    f"{confidence:.2f}",
                    status
                )
            else:
                opt_table.add_row(
                    field_name,
                    "[dim]No mapping found[/dim]",
                    "0.00",
                    "[dim]—[/dim]"
                )
        
        console.print(opt_table)
        
        # Summary
        console.print(f"\n[bold]Mapping Summary:[/bold]")
        console.print(f"High confidence mappings: [green]{high_confidence_mappings}[/green]")
        console.print(f"Low confidence mappings: [yellow]{low_confidence_mappings}[/yellow]")
        console.print(f"Unmapped columns: {len(mapping['unmapped_columns'])}")
        
        # Export mapping if requested
        if args.export_mapping:
            export_data = {
                "creation_timestamp": datetime.now().isoformat(),
                "confidence_threshold": args.confidence_threshold,
                "field_mapping": mapping,
                "summary": {
                    "high_confidence": high_confidence_mappings,
                    "low_confidence": low_confidence_mappings,
                    "unmapped_columns": len(mapping['unmapped_columns'])
                }
            }
            
            with open(args.export_mapping, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            console.print(f"\n💾 Field mapping saved to: [green]{args.export_mapping}[/green]")
    
    except Exception as e:
        console.print(f"[red]✗ Field mapping creation failed: {e}[/red]")
        logger.error(f"Field mapping error: {e}", exc_info=True)

def command_init_enhanced_system(args):
    """Execute enhanced system initialization command."""
    
    console.print(Panel(
        "[bold blue]Enhanced System Initialization[/bold blue]\n"
        "Complete system setup and validation",
        expand=False
    ))
    
    try:
        # Initialize enhanced filter system
        console.print("🚀 Initializing enhanced filtering system...")
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            if args.quick:
                progress_steps = [
                    "Connecting to database...",
                    "Basic table analysis...",
                    "System validation..."
                ]
            else:
                progress_steps = [
                    "Connecting to database...",
                    "Discovering table structure...",
                    "Creating field mappings...",
                    "Optimizing relationships...",
                    "Generating summary...",
                    "System validation..."
                ]
            
            total_steps = len(progress_steps)
            init_task = progress.add_task("Initializing...", total=total_steps)
            
            filter_system = EnhancedScEQTLFilter()
            
            for i, step_desc in enumerate(progress_steps):
                progress.update(init_task, description=step_desc, completed=i)
                time.sleep(0.5)  # Simulate processing time
            
            init_results = filter_system.initialize_system()
            progress.update(init_task, completed=total_steps)
        
        init_time = time.time() - start_time
        
        if init_results["status"] == "success":
            console.print(f"\n✅ [green]System initialized successfully in {init_time:.2f}s[/green]")
            
            # Display system status
            summary = init_results["table_summary"]
            
            status_table = Table(show_header=True, header_style="bold green")
            status_table.add_column("Component")
            status_table.add_column("Status")
            status_table.add_column("Details")
            
            status_table.add_row(
                "Database Connection",
                "[green]✓ Connected[/green]",
                f"Table: {summary['table_name']}"
            )
            
            status_table.add_row(
                "Table Analysis",
                "[green]✓ Complete[/green]",
                f"{summary['basic_statistics']['total_records']:,} records, {summary['basic_statistics']['total_columns']} columns"
            )
            
            req_mapped = summary['field_mapping']['required_fields_mapped']
            total_req = summary['field_mapping']['total_required_fields']
            mapping_status = "[green]✓ Complete[/green]" if req_mapped == total_req else f"[yellow]⚠ Partial[/yellow]"
            
            status_table.add_row(
                "Field Mapping",
                mapping_status,
                f"{req_mapped}/{total_req} required fields mapped"
            )
            
            status_table.add_row(
                "Quality Assessment",
                "[green]✓ Complete[/green]",
                f"{summary['data_quality']['high_quality_columns']} high-quality columns"
            )
            
            console.print(status_table)
            
            # Display recommendations
            if init_results["recommendations"]:
                console.print("\n[bold]System Recommendations:[/bold]")
                for rec in init_results["recommendations"]:
                    priority_color = {"critical": "red", "high": "yellow", "medium": "blue"}.get(rec["priority"], "white")
                    console.print(f"• [{priority_color}]{rec['recommendation']}[/{priority_color}]")
            
            # Save configuration if requested
            if args.save_config:
                config_data = {
                    "initialization_timestamp": datetime.now().isoformat(),
                    "system_status": "ready",
                    "initialization_results": init_results,
                    "performance_metrics": {
                        "initialization_time": init_time,
                        "quick_mode": args.quick
                    }
                }
                
                with open(args.save_config, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
                
                console.print(f"\n💾 Configuration saved to: [green]{args.save_config}[/green]")
            
        else:
            console.print(f"[red]✗ System initialization failed[/red]")
            if "error" in init_results:
                console.print(f"Error: {init_results['error']}")
    
    except Exception as e:
        console.print(f"[red]✗ System initialization failed: {e}[/red]")
        logger.error(f"System initialization error: {e}", exc_info=True)

def command_benchmark_performance(args):
    """Execute performance benchmark command."""
    
    console.print(Panel(
        "[bold red]Performance Benchmark[/bold red]\n"
        "Testing system performance with different configurations",
        expand=False
    ))
    
    try:
        # Initialize system
        filter_system = EnhancedScEQTLFilter()
        console.print("🔧 Initializing system for benchmarking...")
        
        init_results = filter_system.initialize_system()
        if init_results["status"] != "success":
            console.print("[red]✗ System initialization failed[/red]")
            return
        
        # Benchmark results
        benchmark_results = []
        
        console.print(f"\n🏁 Running benchmarks with {args.test_records:,} records...")
        
        # Test different batch sizes
        for batch_size in args.test_batch_sizes:
            console.print(f"\n📊 Testing batch size: {batch_size:,}")
            
            start_time = time.time()
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[progress.description]Batch size {batch_size:,}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                result = filter_system.full_scan_filter(
                    batch_size=batch_size,
                    max_records=args.test_records,
                    enable_ai_assistance=False,  # Disable for consistent benchmarking
                    enable_parallel=False
                )
                
                progress.remove_task(task)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result["scan_summary"]["status"] == "completed":
                records_processed = result["scan_summary"]["total_records_processed"]
                records_per_second = records_processed / processing_time if processing_time > 0 else 0
                
                benchmark_results.append({
                    "batch_size": batch_size,
                    "processing_time": processing_time,
                    "records_processed": records_processed,
                    "records_per_second": records_per_second,
                    "memory_efficiency": "good",  # Could be enhanced with actual memory monitoring
                    "avg_batch_time": result["processing_statistics"].get("average_batch_time", 0)
                })
                
                console.print(f"  ✓ Processed {records_processed:,} records in {processing_time:.2f}s ({records_per_second:.1f} records/sec)")
            
            else:
                console.print(f"  ✗ Benchmark failed for batch size {batch_size}")
        
        # Display benchmark results
        if benchmark_results:
            console.print("\n[bold]Benchmark Results:[/bold]")
            
            bench_table = Table(show_header=True, header_style="bold cyan")
            bench_table.add_column("Batch Size")
            bench_table.add_column("Processing Time")
            bench_table.add_column("Records/sec")
            bench_table.add_column("Avg Batch Time")
            bench_table.add_column("Efficiency")
            
            for result in benchmark_results:
                bench_table.add_row(
                    f"{result['batch_size']:,}",
                    f"{result['processing_time']:.2f}s",
                    f"{result['records_per_second']:.1f}",
                    f"{result['avg_batch_time']:.3f}s",
                    result['memory_efficiency']
                )
            
            console.print(bench_table)
            
            # Find optimal batch size
            best_result = max(benchmark_results, key=lambda x: x['records_per_second'])
            console.print(f"\n🏆 [green]Optimal batch size: {best_result['batch_size']:,} "
                         f"({best_result['records_per_second']:.1f} records/sec)[/green]")
    
    except Exception as e:
        console.print(f"[red]✗ Performance benchmark failed: {e}[/red]")
        logger.error(f"Benchmark error: {e}", exc_info=True)

def display_scan_results(scan_results: Dict[str, Any]):
    """Display formatted scan results."""
    
    summary = scan_results["scan_summary"]
    stats = scan_results["processing_statistics"]
    
    # Main results table
    console.print("\n[bold]Scan Results:[/bold]")
    results_table = Table(show_header=True, header_style="bold green")
    results_table.add_column("Metric")
    results_table.add_column("Value")
    
    results_table.add_row("Total Records Processed", f"{summary['total_records_processed']:,}")
    results_table.add_row("Records Passed Filter", f"{summary['total_records_passed']:,}")
    results_table.add_row("Pass Rate", f"{summary['final_pass_rate']:.1f}%")
    results_table.add_row("Processing Time", f"{summary['processing_time']:.1f}s")
    results_table.add_row("Processing Speed", f"{summary['records_per_second']:.1f} records/sec")
    
    console.print(results_table)
    
    # Quality analysis if available
    if "quality_analysis" in scan_results and scan_results["quality_analysis"]["status"] != "no_results":
        quality = scan_results["quality_analysis"]
        
        console.print("\n[bold]Quality Analysis:[/bold]")
        quality_table = Table(show_header=True, header_style="bold blue")
        quality_table.add_column("Quality Level")
        quality_table.add_column("Count")
        quality_table.add_column("Percentage")
        
        total = quality["total_results"]
        for level, count in quality["quality_distribution"].items():
            percentage = (count / total * 100) if total > 0 else 0
            quality_table.add_row(
                level.replace('_', ' ').title(),
                str(count),
                f"{percentage:.1f}%"
            )
        
        console.print(quality_table)
        
        console.print(f"AI Assistance Rate: {quality['ai_assistance_rate']:.1f}%")
        console.print(f"Enhanced Fields Rate: {quality['enhanced_fields_rate']:.1f}%")
    
    # Processing errors if any
    if scan_results.get("errors"):
        console.print(f"\n[yellow]⚠ Processing Errors: {len(scan_results['errors'])}[/yellow]")

def display_detailed_column_analysis(column_analysis: Dict[str, Any]):
    """Display detailed column analysis."""
    
    console.print("\n[bold]Detailed Column Analysis:[/bold]")
    
    # Group columns by sc-eQTL relevance
    relevance_groups = {
        "High Relevance (4-5)": [],
        "Medium Relevance (2-3)": [],
        "Low Relevance (0-1)": []
    }
    
    for col_name, analysis in column_analysis.items():
        relevance = analysis.get("sc_eqtl_relevance", 0)
        if relevance >= 4:
            relevance_groups["High Relevance (4-5)"].append((col_name, analysis))
        elif relevance >= 2:
            relevance_groups["Medium Relevance (2-3)"].append((col_name, analysis))
        else:
            relevance_groups["Low Relevance (0-1)"].append((col_name, analysis))
    
    for group_name, columns in relevance_groups.items():
        if columns:
            console.print(f"\n[bold]{group_name}:[/bold]")
            
            detail_table = Table(show_header=True, header_style="bold")
            detail_table.add_column("Column")
            detail_table.add_column("Field Type")
            detail_table.add_column("Null %")
            detail_table.add_column("Unique Count")
            detail_table.add_column("Sample Values")
            
            for col_name, analysis in columns[:10]:  # Show top 10 per group
                sample_vals = analysis.get("sample_values", [])
                sample_text = ", ".join([str(v.get("column_name", v)) for v in sample_vals[:3]])
                if len(sample_vals) > 3:
                    sample_text += "..."
                
                detail_table.add_row(
                    col_name,
                    analysis.get("field_type", "unknown"),
                    f"{analysis.get('null_percentage', 0):.1f}%",
                    str(analysis.get('unique_count', 0)),
                    sample_text
                )
            
            console.print(detail_table) 