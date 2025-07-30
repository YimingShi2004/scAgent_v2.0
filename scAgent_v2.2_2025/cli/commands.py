"""
Command handlers for scAgent CLI.
"""

import argparse
import json
import logging
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime

from ..db import (
    test_connection as db_test_connection,
    get_table_info,
    export_schema_report,
    query_geo_master,
    query_sra_master,
    find_scrna_datasets,
    export_query_results,
    get_dataset_statistics
)
from ..models import get_qwen_client, create_analysis_prompt, create_eqtl_evaluation_prompt, create_data_cleaning_prompt

logger = logging.getLogger(__name__)
console = Console()

def test_connection(args: argparse.Namespace) -> int:
    """Test database and model API connections."""
    
    console.print(Panel.fit("Testing Connections", style="bold blue"))
    
    success = True
    
    if not args.model_only:
        # Test database connection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Testing database connection...", total=None)
            
            try:
                db_result = db_test_connection()
                progress.update(task, description="Database connection tested")
                
                if db_result["status"] == "success":
                    console.print("✅ Database connection successful", style="green")
                    console.print(f"   Database: {db_result['database']}")
                    console.print(f"   Host: {db_result['connection_params']['host']}")
                    console.print(f"   Tables found: {len(db_result['tables'])}")
                    
                    # Show available tables
                    if db_result['tables']:
                        table = Table(title="Available Tables")
                        table.add_column("Table Name", style="cyan")
                        for table_name in db_result['tables']:
                            table.add_row(table_name)
                        console.print(table)
                else:
                    console.print("❌ Database connection failed", style="red")
                    console.print(f"   Error: {db_result['error']}")
                    success = False
                    
            except Exception as e:
                console.print(f"❌ Database connection failed: {e}", style="red")
                success = False
    
    if not args.db_only:
        # Test model API connection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Testing model API connection...", total=None)
            
            try:
                client = get_qwen_client()
                model_result = client.test_connection()
                progress.update(task, description="Model API connection tested")
                
                if model_result["status"] == "success":
                    console.print("✅ Model API connection successful", style="green")
                    console.print(f"   Model: {model_result['model']}")
                    console.print(f"   API URL: {model_result['api_url']}")
                    console.print(f"   Response: {model_result['response']}")
                else:
                    console.print("❌ Model API connection failed", style="red")
                    console.print(f"   Error: {model_result['error']}")
                    success = False
                    
            except Exception as e:
                console.print(f"❌ Model API connection failed: {e}", style="red")
                success = False
    
    return 0 if success else 1

def analyze_schema(args: argparse.Namespace) -> int:
    """Analyze database table schemas."""
    
    console.print(Panel.fit("Analyzing Database Schema", style="bold blue"))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing table schemas...", total=None)
            
            table_info = get_table_info(args.tables)
            progress.update(task, description="Schema analysis completed")
        
        # Display results
        for table_name, info in table_info.items():
            if "error" in info:
                console.print(f"❌ Error analyzing {table_name}: {info['error']}", style="red")
                continue
            
            console.print(f"\n📊 Table: {table_name}", style="bold cyan")
            console.print(f"   Rows: {info['row_count']:,}")
            console.print(f"   Columns: {info['column_count']}")
            console.print(f"   Size: {info['table_size']}")
            
            # Show column details
            if info['columns']:
                table = Table(title=f"{table_name} Columns")
                table.add_column("Column", style="cyan")
                table.add_column("Type", style="yellow")
                table.add_column("Nullable", style="magenta")
                
                for col in info['columns'][:10]:  # Show first 10 columns
                    table.add_row(
                        col['column_name'],
                        col['data_type'],
                        "Yes" if col['is_nullable'] == 'YES' else "No"
                    )
                
                if len(info['columns']) > 10:
                    table.add_row("...", "...", "...")
                    
                console.print(table)
        
        # Export report if requested
        if args.output:
            report_file = export_schema_report(args.tables, args.output)
            console.print(f"📄 Schema report exported to: {report_file}", style="green")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Schema analysis failed: {e}", style="red")
        return 1

def analyze_geo(args: argparse.Namespace) -> int:
    """Analyze geo_master table data."""
    
    console.print(Panel.fit("Analyzing GEO Master Data", style="bold blue"))
    
    try:
        # Build query conditions
        conditions = {}
        if args.organisms:
            conditions['organism'] = args.organisms
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Querying geo_master table...", total=None)
            
            # Get data
            results = query_geo_master(
                limit=args.max_records,
                conditions=conditions if conditions else None
            )
            
            # Get statistics
            stats = get_dataset_statistics('geo_master')
            
            progress.update(task, description="Analysis completed")
        
        # Display results
        console.print(f"📊 Found {len(results)} records (limit: {args.max_records})")
        console.print(f"📈 Total records in table: {stats['total_records']:,}")
        
        # Show top organisms
        if 'top_organisms' in stats:
            org_table = Table(title="Top Organisms")
            org_table.add_column("Organism", style="cyan")
            org_table.add_column("Count", style="yellow")
            
            for org in stats['top_organisms'][:5]:
                org_table.add_row(org['organism'], str(org['count']))
            
            console.print(org_table)
        
        # Use AI to analyze the data
        if results:
            console.print("\n🤖 AI Analysis:", style="bold green")
            
            # Prepare data summary for AI
            data_summary = {
                "total_records": len(results),
                "sample_records": results[:3],  # First 3 records
                "statistics": stats
            }
            
            client = get_qwen_client()
            prompt = create_analysis_prompt(
                "Analyze GEO master data for sc-eQTL suitability",
                json.dumps(data_summary, indent=2, default=str)
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running AI analysis...", total=None)
                response = client.generate(prompt, temperature=0.3)
                progress.update(task, description="AI analysis completed")
            
            console.print(Panel(response.content, title="AI Analysis Results"))
        
        # Export results if requested
        if args.output:
            export_file = export_query_results(results, args.output)
            console.print(f"📄 Results exported to: {export_file}", style="green")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ GEO analysis failed: {e}", style="red")
        return 1

def analyze_sra(args: argparse.Namespace) -> int:
    """Analyze sra_master table data."""
    
    console.print(Panel.fit("Analyzing SRA Master Data", style="bold blue"))
    
    try:
        # Build query conditions
        conditions = {}
        if args.platforms:
            conditions['platform'] = args.platforms
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Querying sra_master table...", total=None)
            
            # Get data
            results = query_sra_master(
                limit=args.max_records,
                conditions=conditions if conditions else None
            )
            
            # Get statistics
            stats = get_dataset_statistics('sra_master')
            
            progress.update(task, description="Analysis completed")
        
        # Display results
        console.print(f"📊 Found {len(results)} records (limit: {args.max_records})")
        console.print(f"📈 Total records in table: {stats['total_records']:,}")
        
        # Show top platforms
        if 'top_platforms' in stats:
            platform_table = Table(title="Top Platforms")
            platform_table.add_column("Platform", style="cyan")
            platform_table.add_column("Count", style="yellow")
            
            for platform in stats['top_platforms'][:5]:
                platform_table.add_row(platform['platform'], str(platform['count']))
            
            console.print(platform_table)
        
        # Show library strategies
        if 'library_strategies' in stats:
            strategy_table = Table(title="Library Strategies")
            strategy_table.add_column("Strategy", style="cyan")
            strategy_table.add_column("Count", style="yellow")
            
            for strategy in stats['library_strategies'][:5]:
                strategy_table.add_row(strategy['library_strategy'], str(strategy['count']))
            
            console.print(strategy_table)
        
        # Use AI to analyze the data
        if results:
            console.print("\n🤖 AI Analysis:", style="bold green")
            
            # Prepare data summary for AI
            data_summary = {
                "total_records": len(results),
                "sample_records": results[:3],  # First 3 records
                "statistics": stats
            }
            
            client = get_qwen_client()
            prompt = create_analysis_prompt(
                "Analyze SRA master data for sc-eQTL suitability",
                json.dumps(data_summary, indent=2, default=str)
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running AI analysis...", total=None)
                response = client.generate(prompt, temperature=0.3)
                progress.update(task, description="AI analysis completed")
            
            console.print(Panel(response.content, title="AI Analysis Results"))
        
        # Export results if requested
        if args.output:
            export_file = export_query_results(results, args.output)
            console.print(f"📄 Results exported to: {export_file}", style="green")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ SRA analysis failed: {e}", style="red")
        return 1

def find_eqtl_data(args: argparse.Namespace) -> int:
    """Find datasets suitable for sc-eQTL analysis."""
    
    console.print(Panel.fit("Finding sc-eQTL Suitable Datasets", style="bold blue"))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching for sc-eQTL datasets...", total=None)
            
            # Find datasets
            results = find_scrna_datasets(
                limit=args.max_datasets,
                organisms=args.organisms,
                tissues=args.tissues
            )
            
            progress.update(task, description="Search completed")
        
        # Display results
        console.print(f"🔍 Found {len(results)} potential sc-eQTL datasets")
        
        if results:
            # Show summary table
            summary_table = Table(title="Found Datasets")
            summary_table.add_column("GEO Accession", style="cyan")
            summary_table.add_column("Title", style="yellow", max_width=50)
            summary_table.add_column("Organism", style="green")
            summary_table.add_column("Platform", style="magenta")
            
            for result in results[:10]:  # Show first 10
                summary_table.add_row(
                    result.get('geo_accession', 'N/A'),
                    result.get('title', 'N/A')[:47] + "..." if len(result.get('title', '')) > 50 else result.get('title', 'N/A'),
                    result.get('organism', 'N/A'),
                    result.get('platform', 'N/A')
                )
            
            if len(results) > 10:
                summary_table.add_row("...", "...", "...", "...")
            
            console.print(summary_table)
            
            # Use AI to analyze the datasets
            console.print("\n🤖 AI Analysis:", style="bold green")
            
            # Prepare data summary for AI
            data_summary = {
                "total_datasets": len(results),
                "sample_datasets": results[:5],  # First 5 datasets
                "search_criteria": {
                    "organisms": args.organisms,
                    "tissues": args.tissues,
                    "max_datasets": args.max_datasets
                }
            }
            
            client = get_qwen_client()
            prompt = create_eqtl_evaluation_prompt(
                results[:5],  # Send first 5 datasets for detailed analysis
                data_summary["search_criteria"]
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running AI evaluation...", total=None)
                response = client.generate(prompt, temperature=0.5)
                progress.update(task, description="AI evaluation completed")
            
            console.print(Panel(response.content, title="AI Evaluation Results"))
        
        # Export results if requested
        if args.output:
            export_file = export_query_results(results, args.output, args.format)
            console.print(f"📄 Results exported to: {export_file}", style="green")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ eQTL data search failed: {e}", style="red")
        return 1

def clean_data(args: argparse.Namespace) -> int:
    """Clean and filter table data."""
    
    console.print(Panel.fit(f"Cleaning {args.table} Data", style="bold blue"))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Loading {args.table} data...", total=None)
            
            # Get data based on table
            if args.table == 'geo_master':
                results = query_geo_master(limit=args.max_records)
            else:
                results = query_sra_master(limit=args.max_records)
            
            progress.update(task, description="Data loaded, starting AI cleaning...")
            
            # Use AI to clean and analyze the data
            if results:
                # Prepare data for AI cleaning
                data_summary = {
                    "table_name": args.table,
                    "total_records": len(results),
                    "sample_records": results[:5],  # First 5 records for analysis
                    "all_columns": list(results[0].keys()) if results else []
                }
                
                client = get_qwen_client()
                prompt = create_data_cleaning_prompt(
                    args.table,
                    data_summary,
                    "sc-eQTL"
                )
                
                response = client.generate(prompt, temperature=0.3)
                progress.update(task, description="AI cleaning analysis completed")
                
                console.print(Panel(response.content, title="AI Data Cleaning Analysis"))
            
            # Export the data (for now, export as-is, but with AI recommendations)
            export_file = export_query_results(results, args.output, args.format)
            console.print(f"📄 Data exported to: {export_file}", style="green")
            console.print(f"📊 Exported {len(results)} records", style="blue")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Data cleaning failed: {e}", style="red")
        return 1

def batch_assess(args: argparse.Namespace) -> int:
    """Perform batch quality assessment on datasets."""
    try:
        console.print(Panel.fit("Batch Quality Assessment", style="bold blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading datasets...", total=None)
            
            # Get datasets from specified source
            if args.source == "geo":
                from ..db.query import get_geo_datasets
                results = get_geo_datasets(
                    limit=args.max_records,
                    organisms=args.organisms,
                    tissues=args.tissues
                )
            elif args.source == "sra":
                from ..db.query import get_sra_datasets
                results = get_sra_datasets(
                    limit=args.max_records,
                    organisms=args.organisms,
                    tissues=args.tissues
                )
            else:  # both
                from ..db.query import find_scrna_datasets
                results = find_scrna_datasets(
                    limit=args.max_records,
                    organisms=args.organisms,
                    tissues=args.tissues
                )
            
            progress.update(task, description=f"Loaded {len(results)} datasets")
        
        if not results:
            console.print("❌ No datasets found", style="red")
            return 1
        
        # Perform batch quality assessment
        console.print(f"\n📊 Performing quality assessment on {len(results)} datasets...", style="blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running quality assessment...", total=None)
            
            from ..utils import batch_quality_assessment, detect_batch_effects
            
            # Run quality assessment
            assessment_report = batch_quality_assessment(results, "eqtl")
            
            # Detect batch effects
            batch_effects = detect_batch_effects(results)
            
            progress.update(task, description="Assessment completed")
        
        # Display results
        console.print("\n" + "="*60, style="green")
        console.print("📋 BATCH QUALITY ASSESSMENT RESULTS", style="bold green")
        console.print("="*60, style="green")
        
        # Summary statistics
        summary_table = Table(title="Summary Statistics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        summary_table.add_column("Percentage", style="green")
        
        summary_table.add_row(
            "Total Datasets",
            str(assessment_report["total_records"]),
            "100%"
        )
        summary_table.add_row(
            "High Quality (A/B grade)",
            str(assessment_report["high_quality_count"]),
            f"{assessment_report['high_quality_percentage']:.1f}%"
        )
        summary_table.add_row(
            "Genotype Data Available",
            str(assessment_report["genotype_available_count"]),
            f"{assessment_report['genotype_available_percentage']:.1f}%"
        )
        
        console.print(summary_table)
        
        # Grade distribution
        grade_table = Table(title="Quality Grade Distribution")
        grade_table.add_column("Grade", style="cyan")
        grade_table.add_column("Count", style="yellow")
        grade_table.add_column("Description", style="green")
        
        grade_descriptions = {
            "A": "Excellent - Ready for sc-eQTL",
            "B": "Good - Minor issues",
            "C": "Marginal - Significant limitations",
            "D": "Poor - Major problems"
        }
        
        for grade in ["A", "B", "C", "D"]:
            count = assessment_report["grade_distribution"].get(grade, 0)
            grade_table.add_row(
                grade,
                str(count),
                grade_descriptions[grade]
            )
        
        console.print(grade_table)
        
        # Genotype likelihood distribution
        genotype_table = Table(title="Genotype Data Likelihood")
        genotype_table.add_column("Likelihood", style="cyan")
        genotype_table.add_column("Count", style="yellow")
        genotype_table.add_column("Description", style="green")
        
        genotype_descriptions = {
            "High": "Strong evidence of genotype data",
            "Medium": "Some evidence of genotype data",
            "Low": "No clear evidence of genotype data"
        }
        
        for likelihood in ["High", "Medium", "Low"]:
            count = assessment_report["genotype_distribution"].get(likelihood, 0)
            genotype_table.add_row(
                likelihood,
                str(count),
                genotype_descriptions[likelihood]
            )
        
        console.print(genotype_table)
        
        # Batch effects analysis
        console.print("\n🔬 Batch Effects Analysis", style="bold blue")
        batch_table = Table(title="Batch Effect Risk Assessment")
        batch_table.add_column("Factor", style="cyan")
        batch_table.add_column("Diversity", style="yellow")
        batch_table.add_column("Risk Level", style="green")
        
        batch_table.add_row(
            "Platform Diversity",
            str(batch_effects["platform_diversity"]),
            batch_effects["batch_risk"]
        )
        batch_table.add_row(
            "Temporal Span",
            str(batch_effects["temporal_span"]),
            "Medium" if batch_effects["temporal_span"] > 3 else "Low"
        )
        batch_table.add_row(
            "Lab Diversity",
            str(batch_effects["lab_diversity"]),
            "High" if batch_effects["lab_diversity"] > 5 else "Medium"
        )
        
        console.print(batch_table)
        
        # Top candidates
        if assessment_report["top_candidates"]:
            console.print("\n🏆 Top 5 Candidates for sc-eQTL", style="bold green")
            top_table = Table(title="Highest Quality Datasets")
            top_table.add_column("Accession", style="cyan")
            top_table.add_column("Grade", style="yellow")
            top_table.add_column("Score", style="green")
            top_table.add_column("Organism", style="magenta")
            top_table.add_column("Genotype", style="blue")
            
            for candidate in assessment_report["top_candidates"][:5]:
                top_table.add_row(
                    candidate.get('geo_accession', candidate.get('run_accession', 'N/A')),
                    candidate.get('eqtl_grade', 'N/A'),
                    str(candidate.get('eqtl_score', 0)),
                    candidate.get('organism', 'N/A'),
                    candidate.get('genotype_likelihood', 'N/A')
                )
            
            console.print(top_table)
        
        # Export results if requested
        if args.output:
            # Export detailed assessment
            export_data = {
                "assessment_report": assessment_report,
                "batch_effects": batch_effects,
                "detailed_records": results
            }
            
            import json
            with open(args.output, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            console.print(f"\n📄 Detailed assessment exported to: {args.output}", style="green")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Batch assessment failed: {e}", style="red")
        return 1

def generate_downloads(args: argparse.Namespace) -> int:
    """Generate SRA download commands for selected datasets."""
    try:
        console.print(Panel.fit("SRA Download Script Generation", style="bold blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Finding SRA datasets...", total=None)
            
            # Get SRA datasets
            from ..db.query import get_sra_datasets
            results = get_sra_datasets(
                limit=args.max_records,
                organisms=args.organisms,
                tissues=args.tissues
            )
            
            progress.update(task, description=f"Found {len(results)} SRA datasets")
        
        if not results:
            console.print("❌ No SRA datasets found", style="red")
            return 1
        
        # Filter by quality if requested
        if args.quality_filter:
            console.print("🔍 Applying quality filters...", style="blue")
            
            from ..utils import batch_quality_assessment
            assessment = batch_quality_assessment(results, "eqtl")
            
            # Filter for high quality datasets (A/B grade)
            high_quality = [r for r in assessment["top_candidates"] if r.get('eqtl_grade') in ['A', 'B']]
            
            if high_quality:
                results = high_quality
                console.print(f"✅ Filtered to {len(results)} high-quality datasets", style="green")
            else:
                console.print("⚠️  No high-quality datasets found, using all datasets", style="yellow")
        
        # Generate download commands
        console.print(f"\n📥 Generating download commands for {len(results)} datasets...", style="blue")
        
        from ..utils import generate_sra_download_commands
        download_commands = generate_sra_download_commands(results, args.output_dir)
        
        # Create download script
        script_content = f"""#!/bin/bash
# SRA Download Script
# Generated by scAgent on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# This script downloads {len(results)} SRA datasets suitable for sc-eQTL analysis
# 
# Prerequisites:
# - SRA Toolkit installed (https://github.com/ncbi/sra-tools)
# - Sufficient disk space (estimate ~1-10GB per dataset)
# 
# Usage:
# chmod +x {args.script_name}
# ./{args.script_name}

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "Starting SRA download process..."
echo "Target directory: {args.output_dir}"
echo "Number of datasets: {len(results)}"
echo ""

# Create output directory
{download_commands[0]}

# Download commands
"""
        
        for i, cmd in enumerate(download_commands[1:], 1):
            script_content += f"""
echo "Processing dataset {i}/{len(results) - 1}..."
{cmd}
if [ $? -eq 0 ]; then
    echo "✅ Dataset {i} downloaded successfully"
else
    echo "❌ Dataset {i} download failed"
fi
"""
        
        script_content += f"""
echo ""
echo "Download process completed!"
echo "Check {args.output_dir} for downloaded files"
"""
        
        # Write script to file
        with open(args.script_name, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        import os
        os.chmod(args.script_name, 0o755)
        
        # Display summary
        console.print("\n" + "="*60, style="green")
        console.print("📋 DOWNLOAD SCRIPT GENERATED", style="bold green")
        console.print("="*60, style="green")
        
        summary_table = Table(title="Download Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        
        summary_table.add_row("Total Datasets", str(len(results)))
        summary_table.add_row("Script File", args.script_name)
        summary_table.add_row("Output Directory", args.output_dir)
        summary_table.add_row("Total Commands", str(len(download_commands)))
        
        console.print(summary_table)
        
        # Show sample datasets
        if results:
            console.print("\n📊 Sample Datasets to Download", style="bold blue")
            sample_table = Table(title="First 5 Datasets")
            sample_table.add_column("Run Accession", style="cyan")
            sample_table.add_column("Study Title", style="yellow", max_width=40)
            sample_table.add_column("Organism", style="green")
            sample_table.add_column("Library Strategy", style="magenta")
            
            for result in results[:5]:
                sample_table.add_row(
                    result.get('run_accession', 'N/A'),
                    (result.get('study_title', 'N/A')[:37] + "..." 
                     if len(result.get('study_title', '')) > 40 
                     else result.get('study_title', 'N/A')),
                    result.get('organism', 'N/A'),
                    result.get('library_strategy', 'N/A')
                )
            
            console.print(sample_table)
        
        console.print(f"\n✅ Download script created: {args.script_name}", style="green")
        console.print(f"📁 Files will be downloaded to: {args.output_dir}", style="blue")
        console.print(f"🚀 Run: chmod +x {args.script_name} && ./{args.script_name}", style="yellow")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Download script generation failed: {e}", style="red")
        return 1

def comprehensive_report(args: argparse.Namespace) -> int:
    """Generate comprehensive data quality report."""
    try:
        console.print(Panel.fit("Comprehensive Data Quality Report", style="bold blue"))
        
        # Initialize report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "organisms": args.organisms,
                "tissues": args.tissues,
                "max_records": args.max_records,
                "include_ai_analysis": args.include_ai
            },
            "sections": {}
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # 1. Database connection test
            task = progress.add_task("Testing database connection...", total=None)
            try:
                from ..db.connect import test_connection
                db_status = test_connection()
                report_data["sections"]["database_status"] = {
                    "status": "connected" if db_status else "failed",
                    "details": "Database connection successful" if db_status else "Failed to connect"
                }
            except Exception as e:
                report_data["sections"]["database_status"] = {
                    "status": "error",
                    "details": str(e)
                }
            
            # 2. Schema analysis
            progress.update(task, description="Analyzing database schemas...")
            try:
                from ..db.schema import analyze_table_schema
                
                # Analyze GEO schema
                geo_schema = analyze_table_schema("geo_master")
                sra_schema = analyze_table_schema("sra_master")
                
                report_data["sections"]["schema_analysis"] = {
                    "geo_master": {
                        "rows": geo_schema.get("row_count", 0),
                        "columns": len(geo_schema.get("columns", [])),
                        "size": geo_schema.get("table_size", "Unknown")
                    },
                    "sra_master": {
                        "rows": sra_schema.get("row_count", 0),
                        "columns": len(sra_schema.get("columns", [])),
                        "size": sra_schema.get("table_size", "Unknown")
                    }
                }
            except Exception as e:
                report_data["sections"]["schema_analysis"] = {"error": str(e)}
            
            # 3. Data quality assessment
            progress.update(task, description="Performing data quality assessment...")
            try:
                from ..db.query import find_scrna_datasets
                from ..utils import batch_quality_assessment, detect_batch_effects
                
                # Get datasets
                datasets = find_scrna_datasets(
                    limit=args.max_records,
                    organisms=args.organisms,
                    tissues=args.tissues
                )
                
                if datasets:
                    # Run quality assessment
                    quality_report = batch_quality_assessment(datasets, "eqtl")
                    batch_effects = detect_batch_effects(datasets)
                    
                    report_data["sections"]["quality_assessment"] = {
                        "total_datasets": len(datasets),
                        "high_quality_count": quality_report.get("high_quality_count", 0),
                        "high_quality_percentage": quality_report.get("high_quality_percentage", 0),
                        "genotype_available_count": quality_report.get("genotype_available_count", 0),
                        "grade_distribution": quality_report.get("grade_distribution", {}),
                        "batch_effects": {
                            "platform_diversity": batch_effects.get("platform_diversity", 0),
                            "temporal_span": batch_effects.get("temporal_span", 0),
                            "batch_risk": batch_effects.get("batch_risk", "Unknown")
                        }
                    }
                    
                    # Store top candidates
                    top_candidates = quality_report.get("top_candidates", [])[:10]
                    report_data["sections"]["top_candidates"] = top_candidates
                    
                else:
                    report_data["sections"]["quality_assessment"] = {"error": "No datasets found"}
                    
            except Exception as e:
                report_data["sections"]["quality_assessment"] = {"error": str(e)}
            
            # 4. AI Analysis (if requested)
            if args.include_ai and datasets:
                progress.update(task, description="Running AI analysis...")
                try:
                    from ..models import get_qwen_client, create_eqtl_evaluation_prompt
                    
                    client = get_qwen_client()
                    prompt = create_eqtl_evaluation_prompt(
                        datasets[:5],  # Top 5 datasets
                        {
                            "organisms": args.organisms,
                            "tissues": args.tissues,
                            "max_datasets": args.max_records
                        }
                    )
                    
                    response = client.generate(prompt, temperature=0.5)
                    report_data["sections"]["ai_analysis"] = {
                        "analysis": response.content,
                        "model_used": client.model_name,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    report_data["sections"]["ai_analysis"] = {"error": str(e)}
            
            progress.update(task, description="Report generation completed")
        
        # Display summary
        console.print("\n" + "="*80, style="green")
        console.print("📋 COMPREHENSIVE DATA QUALITY REPORT", style="bold green")
        console.print("="*80, style="green")
        
        # Database status
        db_status = report_data["sections"].get("database_status", {})
        status_color = "green" if db_status.get("status") == "connected" else "red"
        console.print(f"🔗 Database Status: {db_status.get('status', 'unknown')}", style=status_color)
        
        # Schema info
        schema_info = report_data["sections"].get("schema_analysis", {})
        if "error" not in schema_info:
            console.print(f"📊 GEO Records: {schema_info.get('geo_master', {}).get('rows', 0)}", style="blue")
            console.print(f"📊 SRA Records: {schema_info.get('sra_master', {}).get('rows', 0)}", style="blue")
        
        # Quality assessment
        quality_info = report_data["sections"].get("quality_assessment", {})
        if "error" not in quality_info:
            console.print(f"🎯 Total Datasets Analyzed: {quality_info.get('total_datasets', 0)}", style="yellow")
            console.print(f"⭐ High Quality Datasets: {quality_info.get('high_quality_count', 0)} ({quality_info.get('high_quality_percentage', 0):.1f}%)", style="green")
            console.print(f"🧬 Genotype Data Available: {quality_info.get('genotype_available_count', 0)}", style="cyan")
            console.print(f"🔬 Batch Effect Risk: {quality_info.get('batch_effects', {}).get('batch_risk', 'Unknown')}", style="magenta")
        
        # Export report
        import json
        with open(args.output, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        console.print(f"\n📄 Comprehensive report exported to: {args.output}", style="green")
        
        # Display top candidates table
        if report_data["sections"].get("top_candidates"):
            console.print("\n🏆 Top Candidates for sc-eQTL Analysis", style="bold green")
            top_table = Table(title="Highest Quality Datasets")
            top_table.add_column("Rank", style="cyan")
            top_table.add_column("Accession", style="yellow")
            top_table.add_column("Grade", style="green")
            top_table.add_column("Score", style="blue")
            top_table.add_column("Organism", style="magenta")
            
            for i, candidate in enumerate(report_data["sections"]["top_candidates"][:5], 1):
                top_table.add_row(
                    str(i),
                    candidate.get('geo_accession', candidate.get('run_accession', 'N/A')),
                    candidate.get('eqtl_grade', 'N/A'),
                    str(candidate.get('eqtl_score', 0)),
                    candidate.get('organism', 'N/A')
                )
            
            console.print(top_table)
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Comprehensive report generation failed: {e}", style="red")
        return 1

def show_help(args: argparse.Namespace) -> int:
    """Show detailed help and usage examples."""
    console.print(Panel.fit("scAgent - Single-cell RNA-seq Data Agent for eQTL Analysis", style="bold blue"))
    
    console.print("\n🔍 [bold green]Available Commands:[/bold green]")
    
    # Command descriptions
    commands = [
        {
            "name": "test-connection",
            "description": "Test database and model API connections",
            "example": "scagent test-connection",
            "use_case": "Verify system connectivity before analysis"
        },
        {
            "name": "analyze-schema",
            "description": "Analyze database table schemas",
            "example": "scagent analyze-schema --table geo_master",
            "use_case": "Understand database structure and column types"
        },
        {
            "name": "analyze-geo",
            "description": "Analyze GEO master data with AI insights",
            "example": "scagent analyze-geo --max-records 100 --output geo_analysis.csv",
            "use_case": "Explore GEO datasets and get AI-powered analysis"
        },
        {
            "name": "analyze-sra",
            "description": "Analyze SRA master data with AI insights",
            "example": "scagent analyze-sra --max-records 100 --output sra_analysis.csv",
            "use_case": "Explore SRA datasets and get AI-powered analysis"
        },
        {
            "name": "find-eqtl-data",
            "description": "Find datasets suitable for sc-eQTL analysis",
            "example": "scagent find-eqtl-data --organisms 'Homo sapiens' --max-datasets 50 --output eqtl_candidates.csv",
            "use_case": "Identify high-quality datasets for eQTL studies"
        },
        {
            "name": "batch-assess",
            "description": "Perform batch quality assessment on datasets",
            "example": "scagent batch-assess --source both --organisms 'Homo sapiens' --output assessment.json",
            "use_case": "Evaluate data quality across multiple datasets"
        },
        {
            "name": "generate-downloads",
            "description": "Generate SRA download scripts",
            "example": "scagent generate-downloads --organisms 'Homo sapiens' --quality-filter --script-name download.sh",
            "use_case": "Create automated download scripts for selected datasets"
        },
        {
            "name": "clean-data",
            "description": "Clean and filter table data",
            "example": "scagent clean-data --table geo_master --output cleaned_data.csv",
            "use_case": "Standardize and clean dataset metadata"
        },
        {
            "name": "comprehensive-report",
            "description": "Generate comprehensive data quality report",
            "example": "scagent comprehensive-report --organisms 'Homo sapiens' --include-ai --output report.json",
            "use_case": "Create detailed analysis reports for project documentation"
        }
    ]
    
    for cmd in commands:
        console.print(f"\n📋 [bold cyan]{cmd['name']}[/bold cyan]")
        console.print(f"   {cmd['description']}")
        console.print(f"   [dim]Example:[/dim] {cmd['example']}")
        console.print(f"   [dim]Use case:[/dim] {cmd['use_case']}")
    
    console.print("\n🎯 [bold green]Common Workflows:[/bold green]")
    
    workflows = [
        {
            "name": "Quick Start",
            "steps": [
                "scagent test-connection",
                "scagent find-eqtl-data --organisms 'Homo sapiens' --max-datasets 10 --output candidates.csv",
                "scagent batch-assess --source both --organisms 'Homo sapiens' --max-records 50"
            ]
        },
        {
            "name": "Data Download Pipeline",
            "steps": [
                "scagent find-eqtl-data --organisms 'Homo sapiens' --output candidates.csv",
                "scagent generate-downloads --organisms 'Homo sapiens' --quality-filter --script-name download.sh",
                "chmod +x download.sh && ./download.sh"
            ]
        },
        {
            "name": "Comprehensive Analysis",
            "steps": [
                "scagent analyze-geo --max-records 200 --output geo_analysis.csv",
                "scagent analyze-sra --max-records 200 --output sra_analysis.csv",
                "scagent comprehensive-report --organisms 'Homo sapiens' --include-ai --output full_report.json"
            ]
        }
    ]
    
    for workflow in workflows:
        console.print(f"\n🚀 [bold yellow]{workflow['name']}:[/bold yellow]")
        for i, step in enumerate(workflow['steps'], 1):
            console.print(f"   {i}. {step}")
    
    console.print("\n📊 [bold green]Key Features:[/bold green]")
    features = [
        "🔍 AI-powered dataset analysis using Qwen3-235B-A22B model",
        "🧬 Specialized sc-eQTL suitability scoring",
        "📈 Batch quality assessment with genotype data detection",
        "🔬 Batch effect analysis and platform compatibility checks",
        "📥 Automated SRA download script generation",
        "📋 Comprehensive reporting with exportable results",
        "🎯 Multi-organism support (Homo sapiens, Mus musculus, etc.)",
        "🔗 PostgreSQL database integration with robust connection handling"
    ]
    
    for feature in features:
        console.print(f"   {feature}")
    
    console.print("\n⚙️ [bold green]Configuration:[/bold green]")
    console.print("   📁 Config file: scAgent/settings.yml")
    console.print("   🗄️ Database: PostgreSQL at 10.28.1.24:5432")
    console.print("   🤖 AI Model: Qwen3-235B-A22B at http://10.28.1.21:30080/v1")
    console.print("   📊 Tables: geo_master (203 records), sra_master (203 records)")
    
    console.print("\n📚 [bold green]For More Help:[/bold green]")
    console.print("   Use --help with any command for detailed options")
    console.print("   Example: scagent find-eqtl-data --help")
    
    return 0 

def deep_analyze(args: argparse.Namespace) -> int:
    """Perform deep analysis of database table structure and content."""
    try:
        console.print(Panel.fit("Deep Table Structure Analysis", style="bold blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing table structure...", total=None)
            
            from ..db.schema import generate_table_profile
            
            # Generate comprehensive profile
            profile = generate_table_profile(args.table)
            
            progress.update(task, description="Analysis completed")
        
        # Display results
        console.print(f"\n📊 [bold green]Deep Analysis Results for {args.table}[/bold green]")
        console.print("="*80, style="green")
        
        # Basic info
        basic_info = profile["basic_info"]
        console.print(f"📋 Rows: {basic_info['row_count']:,}")
        console.print(f"📋 Columns: {basic_info['column_count']}")
        console.print(f"📋 Size: {basic_info['table_size']}")
        
        # Data quality overview
        quality = profile["data_quality"]
        console.print(f"\n🔍 [bold yellow]Data Quality Overview[/bold yellow]")
        console.print(f"   Columns with nulls: {quality['columns_with_nulls']}")
        console.print(f"   High null columns (>50%): {len(quality['high_null_columns'])}")
        console.print(f"   Low diversity columns: {len(quality['low_diversity_columns'])}")
        
        # sc-eQTL relevance summary
        relevance = profile["sc_eqtl_relevance"]
        console.print(f"\n🧬 [bold cyan]sc-eQTL Relevance Summary[/bold cyan]")
        
        relevance_table = Table(title="Relevant Column Categories")
        relevance_table.add_column("Category", style="cyan")
        relevance_table.add_column("Columns Found", style="yellow")
        relevance_table.add_column("Column Names", style="green")
        
        for category, columns in relevance.items():
            if columns:
                col_names = ", ".join([col["column_name"] for col in columns[:3]])
                if len(columns) > 3:
                    col_names += f" (+{len(columns)-3} more)"
                relevance_table.add_row(
                    category.replace("_", " ").title(),
                    str(len(columns)),
                    col_names
                )
        
        console.print(relevance_table)
        
        # Detailed column analysis
        if args.show_details:
            console.print(f"\n📋 [bold magenta]Detailed Column Analysis[/bold magenta]")
            
            column_details = profile["column_analysis"]["column_details"]
            
            detail_table = Table(title="Column Details")
            detail_table.add_column("Column", style="cyan")
            detail_table.add_column("Type", style="yellow")
            detail_table.add_column("Unique", style="green")
            detail_table.add_column("Null %", style="red")
            detail_table.add_column("Sample Values", style="blue", max_width=40)
            
            for col_name, col_info in column_details.items():
                sample_vals = str(col_info["sample_values"][:3])
                if len(col_info["sample_values"]) > 3:
                    sample_vals = sample_vals[:-1] + ", ...]"
                
                detail_table.add_row(
                    col_name,
                    col_info["data_type"],
                    str(col_info["unique_count"]),
                    f"{col_info['null_percentage']:.1f}%",
                    sample_vals
                )
            
            console.print(detail_table)
        
        # Export results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            console.print(f"\n📄 Detailed analysis exported to: {args.output}", style="green")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Deep analysis failed: {e}", style="red")
        return 1 

def full_export(args: argparse.Namespace) -> int:
    """Export complete table data for server-side processing."""
    try:
        console.print(Panel.fit("Full Table Data Export", style="bold blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading complete table data...", total=None)
            
            from ..db.query import execute_query
            
            # Build query based on table
            if args.table == "geo_master":
                query = "SELECT * FROM geo_master ORDER BY id"
            elif args.table == "sra_master":
                query = "SELECT * FROM sra_master ORDER BY id"
            else:
                raise ValueError(f"Unsupported table: {args.table}")
            
            # Execute query to get all data
            results = execute_query(query)
            
            progress.update(task, description=f"Loaded {len(results)} records")
        
        if not results:
            console.print("❌ No data found in table", style="red")
            return 1
        
        # Display summary
        console.print(f"\n📊 [bold green]Full Export Results[/bold green]")
        console.print("="*60, style="green")
        console.print(f"📋 Table: {args.table}")
        console.print(f"📋 Total Records: {len(results):,}")
        console.print(f"📋 Columns: {len(results[0].keys()) if results else 0}")
        
        # Show sample of data
        if args.show_sample and results:
            console.print(f"\n📋 [bold yellow]Sample Data (First 5 Records)[/bold yellow]")
            
            # Create sample table
            sample_table = Table(title="Sample Records")
            
            # Add columns (limit to first 8 for display)
            columns = list(results[0].keys())[:8]
            for col in columns:
                sample_table.add_column(col, style="cyan", max_width=20)
            
            # Add sample rows
            for i, record in enumerate(results[:5]):
                row_data = []
                for col in columns:
                    value = str(record.get(col, ''))
                    if len(value) > 18:
                        value = value[:15] + "..."
                    row_data.append(value)
                sample_table.add_row(*row_data)
            
            console.print(sample_table)
            
            if len(columns) > 8:
                console.print(f"   ... and {len(results[0].keys()) - 8} more columns")
        
        # Export data
        console.print(f"\n📥 [bold blue]Exporting data...[/bold blue]")
        
        if args.format == "csv":
            import csv
            with open(args.output, 'w', newline='', encoding='utf-8') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        
        elif args.format == "json":
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        elif args.format == "parquet":
            try:
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_parquet(args.output, index=False)
            except ImportError:
                console.print("❌ Parquet format requires pandas and pyarrow. Using CSV instead.", style="yellow")
                args.output = args.output.replace('.parquet', '.csv')
                import csv
                with open(args.output, 'w', newline='', encoding='utf-8') as f:
                    if results:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
        
        console.print(f"✅ Data exported to: {args.output}", style="green")
        console.print(f"📊 Exported {len(results):,} records", style="blue")
        
        # Generate metadata file if requested
        if args.include_metadata:
            import json
            
            metadata = {
                "export_timestamp": datetime.now().isoformat(),
                "table_name": args.table,
                "total_records": len(results),
                "total_columns": len(results[0].keys()) if results else 0,
                "column_names": list(results[0].keys()) if results else [],
                "export_format": args.format,
                "export_parameters": {
                    "show_sample": args.show_sample,
                    "include_metadata": args.include_metadata
                }
            }
            
            metadata_file = args.output.rsplit('.', 1)[0] + '_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            console.print(f"📄 Metadata exported to: {metadata_file}", style="cyan")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Full export failed: {e}", style="red")
        return 1 

def comprehensive_clean(args: argparse.Namespace) -> int:
    """Comprehensive data cleaning and filtering for sc-eQTL analysis (优化版)."""
    try:
        # Apply safe string handling patch
        from ..utils_safe import patch_utils_safe_strings
        patch_utils_safe_strings()
        
        # Apply improved assessment functions
        from ..utils_improved_assessment import patch_assessment_functions
        patch_assessment_functions()
        
        # Apply enhanced species detection
        from ..utils_species_enhanced import patch_species_assessment
        patch_species_assessment()
        
        console.print(Panel.fit("Comprehensive Data Cleaning & Filtering (Optimized)", style="bold blue"))
        
        # Load configuration
        filter_config = {
            "required_species": args.species if args.species else ["Homo sapiens"],
            "exclude_cell_lines": not args.include_cell_lines,
            "require_database_id": True,
            "require_publication": args.require_publication,
            "require_sample_size": args.require_sample_size,
            "require_country_info": args.require_country,
            "require_age_info": args.require_age,
            "require_tumor_annotation": args.require_tumor_status,
            "require_sequencing_method": True,
            "require_tissue_source": True
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: 字段智能映射
            from ..db.merged_table_handler import MergedTableHandler
            handler = MergedTableHandler()
            column_mapping = handler.create_column_mapping()
            
            # Step 2: Load data
            task = progress.add_task("Loading data from tables...", total=None)
            from ..db.query import execute_query
            all_records = []
            # Load SRA data (primary source)
            if args.include_sra:
                sra_limit = ""
                if hasattr(args, 'limit') and args.limit:
                    sra_limit = f"LIMIT {args.limit}"
                sra_query = f"""
                SELECT * FROM srameta.sra_master WHERE "run_accession" IS NOT NULL AND "run_accession" != '' ORDER BY "sra_ID" DESC {sra_limit}
                """
                sra_records = execute_query(sra_query)
                all_records.extend(sra_records)
                progress.update(task, description=f"Loaded {len(sra_records)} SRA records (primary)")
            # Load GEO data (supplementary source)
            if args.include_geo:
                geo_limit = ""
                if hasattr(args, 'limit') and args.limit:
                    geo_limit_num = min(args.limit // 10, 1000)
                    geo_limit = f"LIMIT {geo_limit_num}"
                geo_query = f"""
                SELECT * FROM geometa.geo_master ORDER BY "gse_ID" DESC {geo_limit}
                """
                geo_records = execute_query(geo_query)
                all_records.extend(geo_records)
                progress.update(task, description=f"Loaded {len(geo_records)} GEO records (supplementary)")
                progress.update(task, description=f"Loaded {len(all_records)} total records")
            # Step 3: Apply enhanced pre-filtering for human single-cell data
            progress.update(task, description="Applying enhanced pre-filtering for human single-cell data...")
            from ..utils_enhanced_filtering import (
                pre_filter_record, is_human_sample_enhanced, is_single_cell_experiment,
                extract_instrument_model, extract_sequencing_strategy, extract_sc_eqtl_criteria_with_ai
            )
            
            pre_filtered_records = []
            rejection_stats = {"Not a human sample": 0, "Sample is a cell line": 0, "Not a single-cell experiment": 0}
            
            for record in all_records:
                rejection_reason = pre_filter_record(record, column_mapping)
                if rejection_reason is None:
                    pre_filtered_records.append(record)
                else:
                    rejection_stats[rejection_reason] += 1
            
            progress.update(task, description=f"Pre-filtering completed: {len(pre_filtered_records)}/{len(all_records)} records passed")
            
            # Log pre-filtering results
            console.print(f"\n🔍 [bold yellow]Pre-filtering Results[/bold yellow]")
            console.print(f"   Original records: {len(all_records):,}")
            console.print(f"   Passed pre-filter: {len(pre_filtered_records):,}")
            console.print(f"   Rejection breakdown:")
            for reason, count in rejection_stats.items():
                console.print(f"     • {reason}: {count:,}")
            
            # Step 4: Apply sc-eQTL filters on pre-filtered data
            progress.update(task, description="Applying sc-eQTL filters on pre-filtered data...")
            from ..utils import apply_sc_eqtl_filters_with_ai, generate_filter_report
            from ..utils import build_geo_sra_mapping, create_integrated_dataset_table
            from ..utils_fixed import create_integrated_dataset_table_fixed
            
            # Separate GEO and SRA records from pre-filtered data
            geo_records = [r for r in pre_filtered_records if r.get('data_source') == 'GEO']
            sra_records = [r for r in pre_filtered_records if r.get('data_source') == 'SRA']
            
            # Build GEO-SRA mapping
            progress.update(task, description="Building GEO-SRA relationships...")
            mapping = build_geo_sra_mapping(geo_records, sra_records)
            
            # Step 5: 创建集成表时传递column_mapping
            progress.update(task, description="Creating integrated dataset table...")
            integrated_table = create_integrated_dataset_table_fixed(
                geo_records, sra_records, mapping, column_mapping=column_mapping
            )
            
            # Step 6: 过滤时传递column_mapping
            use_ai = args.use_ai and not args.no_ai
            progress.update(task, description=f"Applying {'intelligent AI-assisted ' if use_ai else 'intelligent '}sc-eQTL filters...")
            from ..utils import apply_intelligent_sc_eqtl_filters
            filtered_records = apply_intelligent_sc_eqtl_filters(
                integrated_table, 
                filter_config, 
                use_ai=use_ai,
                ai_batch_size=args.ai_batch_size,
                column_mapping=column_mapping
            )
            # Step 6: Generate filter report
            progress.update(task, description="Generating filter report...")
            from ..utils_report_fixed import generate_filter_report_fixed
            filter_report = generate_filter_report_fixed(integrated_table, filtered_records)
            progress.update(task, description="Cleaning completed")
        # Display results
        console.print(f"\n📊 [bold green]Comprehensive Cleaning Results (Optimized)[/bold green]")
        console.print("="*80, style="green")
        # Summary statistics
        summary = filter_report["filtering_summary"]
        console.print(f"📋 Original Records: {summary['original_count']:,}")
        console.print(f"📋 Filtered Records: {summary['filtered_count']:,}")
        console.print(f"📋 Retention Rate: {summary['retention_rate']:.1f}%")
        # GEO-SRA mapping statistics
        mapping_stats = mapping["mapping_stats"]
        console.print(f"\n🔗 [bold blue]GEO-SRA Mapping Statistics[/bold blue]")
        console.print(f"   GEO Records: {mapping_stats['total_geo_records']:,}")
        console.print(f"   SRA Records: {mapping_stats['total_sra_records']:,}")
        console.print(f"   Mapped GEO Records: {mapping_stats['mapped_geo_records']:,} ({mapping_stats['geo_mapping_rate']:.1f}%)")
        console.print(f"   Mapped SRA Records: {mapping_stats['mapped_sra_records']:,} ({mapping_stats['sra_mapping_rate']:.1f}%)")
        console.print(f"   Orphaned GEO Records: {mapping_stats['orphaned_geo_records']:,}")
        console.print(f"   Orphaned SRA Records: {mapping_stats['orphaned_sra_records']:,}")
        # Relationship types
        rel_types = mapping_stats.get('relationship_types', {})
        if rel_types:
            console.print(f"\n📊 [bold magenta]Relationship Types[/bold magenta]")
            console.print(f"   One-to-One: {rel_types.get('one_to_one', 0)}")
            console.print(f"   One-to-Many: {rel_types.get('one_to_many', 0)}")
            console.print(f"   Many-to-One: {rel_types.get('many_to_one', 0)}")
            console.print(f"   Many-to-Many: {rel_types.get('many_to_many', 0)}")
        # Filter statistics
        if filter_report["filter_statistics"]:
            console.print(f"\n🔍 [bold yellow]Filter Performance[/bold yellow]")
            filter_table = Table(title="Filter Statistics")
            filter_table.add_column("Filter", style="cyan")
            filter_table.add_column("Pass Rate", style="green")
            filter_table.add_column("Pass Count", style="yellow")
            filter_table.add_column("Fail Count", style="red")
            for filter_name, stats in filter_report["filter_statistics"].items():
                filter_table.add_row(
                    filter_name.replace("_", " ").title(),
                    f"{stats['pass_rate']:.1f}%",
                    str(stats['pass_count']),
                    str(stats['fail_count'])
                )
            console.print(filter_table)
        # Quality distribution
        if filter_report["quality_distribution"]:
            quality = filter_report["quality_distribution"]
            console.print(f"\n⭐ [bold cyan]Quality Distribution[/bold cyan]")
            console.print(f"   Average Score: {quality['average_score']:.1f}")
            console.print(f"   High Quality (≥8): {quality['high_quality_count']}")
            console.print(f"   Medium Quality (4-7): {quality['medium_quality_count']}")
            console.print(f"   Low Quality (<4): {quality['low_quality_count']}")
        # Common rejection reasons
        if filter_report["common_rejection_reasons"]:
            console.print(f"\n❌ [bold red]Common Rejection Reasons[/bold red]")
            for reason in filter_report["common_rejection_reasons"][:5]:
                console.print(f"   • {reason['reason']}: {reason['count']} records")
        # Prepare cleaned data for export with AI client for enhanced extraction
        ai_client = None
        if use_ai:
            try:
                from ..models import get_qwen_client
                ai_client = get_qwen_client()
            except Exception as e:
                console.print(f"⚠️ AI client unavailable, using rule-based extraction: {e}", style="yellow")
        
        cleaned_data = []
        for record in filtered_records:
            filter_result = record.get("sc_eqtl_filter_result", {})
            
            # Extract 10 key sc-eQTL criteria using AI
            sc_eqtl_criteria = extract_sc_eqtl_criteria_with_ai(record, column_mapping, ai_client)
            
            cleaned_record = {
                # Basic identifiers
                "geo_accession": record.get("geo_accession", ""),
                "sra_run_accession": record.get("sra_run_accession", ""),
                "sra_study_accession": record.get("sra_study_accession", ""),
                "sra_experiment_accession": record.get("sra_experiment_accession", ""),
                "relationship_type": record.get("relationship_type", ""),
                "mapping_confidence": record.get("mapping_confidence", 0.0),
                
                # Complete SRX data information
                "srx_accession": record.get("experiment_accession", ""),
                "srr_accession": record.get("run_accession", ""),
                "srs_accession": record.get("sample_accession", ""),
                "srp_accession": record.get("study_accession", ""),
                "biosample_accession": record.get("biosample_accession", ""),
                "bioproject_accession": record.get("bioproject_accession", ""),
                "library_name": record.get("library_name", ""),
                "library_strategy": record.get("library_strategy", ""),
                "library_source": record.get("library_source", ""),
                "library_selection": record.get("library_selection", ""),
                "library_layout": record.get("library_layout", ""),
                "platform": record.get("platform", ""),
                "instrument_model": record.get("instrument_model", ""),
                "read_count": record.get("spot_count", 0),
                "base_count": record.get("base_count", 0),
                "avg_read_length": record.get("avg_read_length", 0),
                "fastq_ftp_urls": record.get("fastq_ftp", ""),
                "sra_ftp_url": record.get("sra_ftp", ""),
                "submission_date": record.get("submission_date", ""),
                "publication_date": record.get("publication_date", ""),
                "last_update_date": record.get("last_update_date", ""),
                # GEO information
                "geo_title": record.get("geo_title", ""),
                "geo_summary": record.get("geo_summary", ""),
                "geo_organism": record.get("geo_organism", ""),
                "geo_platform": record.get("geo_platform", ""),
                "geo_sample_count": record.get("geo_sample_count", 0),
                "geo_submission_date": record.get("geo_submission_date", ""),
                "geo_status": record.get("geo_status", ""),
                # SRA information
                "sra_study_title": record.get("sra_study_title", ""),
                "sra_organism": record.get("sra_organism", ""),
                "sra_platform": record.get("sra_platform", ""),
                "sra_instrument": record.get("sra_instrument", ""),
                "sra_library_strategy": record.get("sra_library_strategy", ""),
                "sra_library_layout": record.get("sra_library_layout", ""),
                "sra_spots": record.get("sra_spots", 0),
                "sra_bases": record.get("sra_bases", 0),
                # Download information
                "fastq_download_url": record.get("fastq_download_url", ""),
                "fastq_dump_command": record.get("fastq_dump_command", ""),
                "prefetch_command": record.get("prefetch_command", ""),
                "fasterq_dump_command": record.get("fasterq_dump_command", ""),
                "aspera_download_command": record.get("aspera_download_command", ""),
                "estimated_file_size_gb": record.get("estimated_file_size_gb", 0),
                # Quality and recommendation
                "data_completeness": record.get("data_completeness", 0.0),
                "recommended_for_eqtl": record.get("recommended_for_eqtl", ""),
                # Extracted information
                "tissue": extract_tissue_from_text(record),
                "cell_type": extract_cell_type_from_text(record),
                "sequencing_method": extract_sequencing_method(record),
                "publication_info": extract_publication_info(record),
                "country": extract_country_info(record),
                "age_info": extract_age_info(record),
                "tumor_status": extract_tumor_status(record),
                # Enhanced extraction for single-cell data
                "instrument_model_enhanced": extract_instrument_model(record, column_mapping),
                "sequencing_strategy_enhanced": extract_sequencing_strategy(record, column_mapping),
                "is_human_confirmed": is_human_sample_enhanced(record, column_mapping),
                "is_single_cell_confirmed": is_single_cell_experiment(record, column_mapping),
                # AI-extracted 10 key sc-eQTL criteria
                "extracted_organism": sc_eqtl_criteria["organism"],
                "extracted_tissue_type": sc_eqtl_criteria["tissue_type"],
                "extracted_cell_type": sc_eqtl_criteria["cell_type"],
                "extracted_sample_size": sc_eqtl_criteria["sample_size"],
                "extracted_platform": sc_eqtl_criteria["platform"],
                "extracted_project_id": sc_eqtl_criteria["project_id"],
                "extracted_publication": sc_eqtl_criteria["publication"],
                "extracted_geographic_location": sc_eqtl_criteria["geographic_location"],
                "extracted_age_range": sc_eqtl_criteria["age_range"],
                "extracted_disease_status": sc_eqtl_criteria["disease_status"],
                # Quality scores
                "sc_eqtl_overall_score": filter_result.get("overall_score", 0),
                "sc_eqtl_grade": calculate_grade_from_score(filter_result.get("overall_score", 0)),
                "passes_required_filters": filter_result.get("passes_required_filters", False),
                "passes_optional_filters": filter_result.get("passes_optional_filters", False),
                # Individual filter scores
                "species_score": filter_result.get("filter_scores", {}).get("species", 0),
                "cell_line_score": filter_result.get("filter_scores", {}).get("cell_line", 0),
                "database_id_score": filter_result.get("filter_scores", {}).get("database_id", 0),
                "publication_score": filter_result.get("filter_scores", {}).get("publication", 0),
                "sample_size_score": filter_result.get("filter_scores", {}).get("sample_size", 0),
                "country_score": filter_result.get("filter_scores", {}).get("country", 0),
                "age_score": filter_result.get("filter_scores", {}).get("age", 0),
                "tumor_score": filter_result.get("filter_scores", {}).get("tumor", 0),
                "sequencing_method_score": filter_result.get("filter_scores", {}).get("sequencing_method", 0),
                "tissue_score": filter_result.get("filter_scores", {}).get("tissue", 0),
                # Filter reasons
                "filter_reasons": "; ".join(filter_result.get("filter_reasons", [])),
                # Processing metadata
                "processing_timestamp": datetime.now().isoformat(),
                "filter_config_used": str(filter_config)
            }
            cleaned_data.append(cleaned_record)
        # 输出前打印部分高分样本供人工核查
        high_quality = [rec for rec in cleaned_data if rec.get('sc_eqtl_grade') == 'A']
        if high_quality:
            console.print(f"\n[人工核查] Top 3 High-Quality Samples:")
            for rec in high_quality[:3]:
                console.print(rec)
        # Export cleaned data
        console.print(f"\n📥 [bold blue]Exporting cleaned data...[/bold blue]")
        if args.format == "csv":
            import csv
            with open(args.output, 'w', newline='', encoding='utf-8') as f:
                if cleaned_data:
                    writer = csv.DictWriter(f, fieldnames=cleaned_data[0].keys())
                    writer.writeheader()
                    writer.writerows(cleaned_data)
        elif args.format == "json":
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, default=str, ensure_ascii=False)
        console.print(f"✅ Cleaned data exported to: {args.output}", style="green")
        console.print(f"📊 Exported {len(cleaned_data):,} high-quality records", style="blue")
        # Export filter report
        if args.include_report:
            import json
            report_file = args.output.rsplit('.', 1)[0] + '_filter_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(filter_report, f, indent=2, default=str)
            console.print(f"📄 Filter report exported to: {report_file}", style="cyan")
        return 0
    except Exception as e:
        console.print(f"❌ Comprehensive cleaning failed: {e}", style="red")
        return 1

def extract_tissue_from_text(record: Dict[str, Any]) -> str:
    """Extract tissue information from text fields."""
    tissue_keywords = [
        'brain', 'liver', 'heart', 'lung', 'kidney', 'muscle', 'blood',
        'skin', 'bone', 'pancreas', 'stomach', 'intestine', 'colon',
        'breast', 'ovary', 'testis', 'prostate', 'thyroid', 'spleen'
    ]
    
    text_fields = [
        (record.get('title') or '').lower(),
        (record.get('summary') or '').lower(),
        (record.get('study_title') or '').lower()
    ]
    
    for text in text_fields:
        for tissue in tissue_keywords:
            if tissue in text:
                return tissue.title()
    
    return ""

def extract_cell_type_from_text(record: Dict[str, Any]) -> str:
    """Extract cell type information from text fields."""
    cell_type_keywords = [
        'neuron', 'hepatocyte', 'cardiomyocyte', 'fibroblast', 'endothelial',
        'epithelial', 'immune', 'stem cell', 'b cell', 't cell', 'macrophage'
    ]
    
    text_fields = [
        (record.get('title') or '').lower(),
        (record.get('summary') or '').lower(),
        (record.get('study_title') or '').lower()
    ]
    
    for text in text_fields:
        for cell_type in cell_type_keywords:
            if cell_type in text:
                return cell_type.title()
    
    return ""

def extract_sequencing_method(record: Dict[str, Any]) -> str:
    """Extract sequencing method information."""
    methods = [
        '10x genomics', '10x', 'smart-seq', 'smart-seq2', 'drop-seq',
        'cel-seq', 'mars-seq', 'scrb-seq'
    ]
    
    text_fields = [
        (record.get('title') or '').lower(),
        (record.get('summary') or '').lower(),
        (record.get('library_strategy') or '').lower()
    ]
    
    for text in text_fields:
        for method in methods:
            if method in text:
                return method
    
    return ""

def extract_publication_info(record: Dict[str, Any]) -> str:
    """Extract publication information."""
    pub_fields = [
        record.get('pmid', ''),
        record.get('doi', ''),
        record.get('pubmed_id', '')
    ]
    
    pub_info = [field for field in pub_fields if field]
    return "; ".join(pub_info) if pub_info else ""

def extract_country_info(record: Dict[str, Any]) -> str:
    """Extract country information."""
    country_keywords = {
        'usa': 'United States',
        'united states': 'United States',
        'china': 'China',
        'uk': 'United Kingdom',
        'united kingdom': 'United Kingdom',
        'germany': 'Germany',
        'japan': 'Japan',
        'france': 'France',
        'canada': 'Canada'
    }
    
    text_fields = [
        (record.get('title') or '').lower(),
        (record.get('summary') or '').lower(),
        (record.get('country') or '').lower()
    ]
    
    for text in text_fields:
        for keyword, country in country_keywords.items():
            if keyword in text:
                return country
    
    return ""

def extract_age_info(record: Dict[str, Any]) -> str:
    """Extract age information."""
    import re
    
    text_fields = [
        (record.get('title') or '').lower(),
        (record.get('summary') or '').lower(),
        (record.get('characteristics') or '').lower()
    ]
    
    for text in text_fields:
        # Look for age patterns
        age_patterns = [
            r'(\d+)\s*(?:years?\s*old|y\.?o\.?|yr\.?s?)',
            r'(?:age|aged)\s*(\d+)',
            r'(\d+)\s*-\s*(\d+)\s*(?:years?|y\.?o\.?)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
    
    return ""

def extract_tumor_status(record: Dict[str, Any]) -> str:
    """Extract tumor status information."""
    tumor_keywords = ['tumor', 'cancer', 'carcinoma', 'malignant']
    normal_keywords = ['normal', 'healthy', 'control', 'non-tumor']
    
    text_fields = [
        (record.get('title') or '').lower(),
        (record.get('summary') or '').lower(),
        (record.get('disease') or '').lower()
    ]
    
    for text in text_fields:
        for keyword in tumor_keywords:
            if keyword in text:
                return "Tumor"
        for keyword in normal_keywords:
            if keyword in text:
                return "Normal"
    
    return "Unknown"

def calculate_grade_from_score(score: int) -> str:
    """Calculate grade from overall score."""
    if score >= 10:
        return "A"
    elif score >= 7:
        return "B"
    elif score >= 4:
        return "C"
    else:
        return "D" 