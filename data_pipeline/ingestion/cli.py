"""Command-line interface for data ingestion service."""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from typing import List, Optional

from .service import DataIngestionService, IngestionConfig
from .circuit_breaker import CircuitBreakerConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_symbols(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols string."""
    return [s.strip().upper() for s in symbols_str.split(',') if s.strip()]


async def run_ingestion(args):
    """Run data ingestion with CLI arguments."""
    # Create circuit breaker config
    cb_config = CircuitBreakerConfig(
        failure_threshold=args.cb_failure_threshold,
        recovery_timeout=args.cb_recovery_timeout,
        success_threshold=args.cb_success_threshold,
        timeout=args.cb_timeout
    )
    
    # Create ingestion config
    config = IngestionConfig(
        api_provider=args.provider,
        api_key=args.api_key,
        symbols=parse_symbols(args.symbols),
        days_to_fetch=args.days,
        storage_base_path=args.storage_path,
        circuit_breaker_config=cb_config,
        enable_validation=args.enable_validation,
        enable_events=args.enable_events
    )
    
    print(f"Starting data ingestion with configuration:")
    print(f"  Provider: {config.api_provider}")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Days to fetch: {config.days_to_fetch}")
    print(f"  Storage path: {config.storage_base_path}")
    print(f"  Validation enabled: {config.enable_validation}")
    print(f"  Events enabled: {config.enable_events}")
    print()
    
    # Run ingestion
    async with DataIngestionService(config) as service:
        result = await service.run_ingestion_pipeline()
        
        # Print results
        print("Ingestion Results:")
        print("=" * 50)
        print(f"Success: {result['success']}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        print(f"Records fetched: {result['records_fetched']}")
        print(f"Records stored: {result['records_stored']}")
        print(f"Event published: {result['event_published']}")
        
        if result['validation_result']:
            val_result = result['validation_result']
            print(f"Validation: {val_result['status']} ({val_result['passed_checks']}/{val_result['total_checks']} checks passed)")
            
            if val_result['warnings']:
                print("Warnings:")
                for warning in val_result['warnings']:
                    print(f"  - {warning}")
        
        if result['errors']:
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        print()
        
        # Print storage stats
        storage_stats = service.get_storage_stats()
        if storage_stats['total_files'] > 0:
            print("Storage Statistics:")
            print(f"  Total files: {storage_stats['total_files']}")
            print(f"  Total size: {storage_stats['total_size_mb']:.2f} MB")
            print(f"  Symbols: {', '.join(storage_stats['symbols'])}")
            print(f"  Date range: {storage_stats['date_range']['earliest']} to {storage_stats['date_range']['latest']}")
        
        # Print circuit breaker stats
        cb_stats = service.get_circuit_breaker_stats()
        if cb_stats:
            print("Circuit Breaker Statistics:")
            print(f"  State: {cb_stats['state']}")
            print(f"  Failure count: {cb_stats['failure_count']}")
            print(f"  Success count: {cb_stats['success_count']}")
        
        return result['success']


async def list_data(args):
    """List available data."""
    config = IngestionConfig(storage_base_path=args.storage_path)
    
    async with DataIngestionService(config) as service:
        available_data = service.list_available_data()
        
        if not available_data:
            print("No data available in storage.")
            return
        
        print("Available Data:")
        print("=" * 50)
        
        for symbol, dates in available_data.items():
            print(f"{symbol}: {len(dates)} days")
            if args.verbose:
                for date in dates[-10:]:  # Show last 10 dates
                    print(f"  - {date}")
                if len(dates) > 10:
                    print(f"  ... and {len(dates) - 10} more")
            else:
                print(f"  Latest: {dates[-1] if dates else 'None'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QuantFolio Data Ingestion Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data for AAPL and GOOGL using Yahoo Finance
  python -m data_pipeline.ingestion.cli ingest --symbols AAPL,GOOGL --days 5
  
  # Use Alpha Vantage with API key
  python -m data_pipeline.ingestion.cli ingest --provider alpha_vantage --api-key YOUR_KEY --symbols AAPL
  
  # List available data
  python -m data_pipeline.ingestion.cli list --verbose
        """
    )
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Run data ingestion')
    ingest_parser.add_argument('--provider', choices=['yahoo', 'alpha_vantage'], 
                              default='yahoo', help='API provider')
    ingest_parser.add_argument('--api-key', default='', help='API key (required for Alpha Vantage)')
    ingest_parser.add_argument('--symbols', default='AAPL,GOOGL,MSFT,AMZN,TSLA',
                              help='Comma-separated list of symbols')
    ingest_parser.add_argument('--days', type=int, default=1, help='Number of days to fetch')
    ingest_parser.add_argument('--storage-path', default='data', help='Storage base path')
    ingest_parser.add_argument('--no-validation', dest='enable_validation', 
                              action='store_false', help='Disable data validation')
    ingest_parser.add_argument('--no-events', dest='enable_events', 
                              action='store_false', help='Disable event publishing')
    
    # Circuit breaker options
    ingest_parser.add_argument('--cb-failure-threshold', type=int, default=3,
                              help='Circuit breaker failure threshold')
    ingest_parser.add_argument('--cb-recovery-timeout', type=float, default=60.0,
                              help='Circuit breaker recovery timeout (seconds)')
    ingest_parser.add_argument('--cb-success-threshold', type=int, default=2,
                              help='Circuit breaker success threshold')
    ingest_parser.add_argument('--cb-timeout', type=float, default=30.0,
                              help='Circuit breaker request timeout (seconds)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available data')
    list_parser.add_argument('--storage-path', default='data', help='Storage base path')
    list_parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.log_level)
    
    try:
        if args.command == 'ingest':
            success = asyncio.run(run_ingestion(args))
            return 0 if success else 1
        elif args.command == 'list':
            asyncio.run(list_data(args))
            return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Unexpected error occurred")
        return 1


if __name__ == '__main__':
    exit(main())