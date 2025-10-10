"""Example usage of the data ingestion service."""

import asyncio
import logging
from datetime import datetime
from data_pipeline.ingestion.service import DataIngestionService, IngestionConfig
from data_pipeline.ingestion.circuit_breaker import CircuitBreakerConfig


async def basic_ingestion_example():
    """Basic example of data ingestion."""
    print("=== Basic Data Ingestion Example ===")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = IngestionConfig(
        api_provider="yahoo",  # Use Yahoo Finance (no API key required)
        symbols=["AAPL", "GOOGL", "MSFT"],
        days_to_fetch=2,
        storage_base_path="example_data"
    )
    
    # Run ingestion
    async with DataIngestionService(config) as service:
        print(f"Fetching {config.days_to_fetch} days of data for {', '.join(config.symbols)}")
        
        result = await service.run_ingestion_pipeline()
        
        print(f"\nResults:")
        print(f"  Success: {result['success']}")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")
        print(f"  Records fetched: {result['records_fetched']}")
        print(f"  Records stored: {result['records_stored']}")
        
        if result['errors']:
            print(f"  Errors: {result['errors']}")
        
        # Show storage statistics
        storage_stats = service.get_storage_stats()
        print(f"\nStorage Statistics:")
        print(f"  Total files: {storage_stats['total_files']}")
        print(f"  Total size: {storage_stats['total_size_mb']:.2f} MB")
        print(f"  Symbols: {', '.join(storage_stats['symbols'])}")


async def advanced_ingestion_example():
    """Advanced example with custom configuration."""
    print("\n=== Advanced Data Ingestion Example ===")
    
    # Custom circuit breaker configuration
    cb_config = CircuitBreakerConfig(
        failure_threshold=2,  # Open after 2 failures
        recovery_timeout=30.0,  # Try recovery after 30 seconds
        success_threshold=1,  # Close after 1 success
        timeout=15.0  # 15 second request timeout
    )
    
    # Advanced configuration
    config = IngestionConfig(
        api_provider="yahoo",
        symbols=["AAPL", "TSLA", "NVDA"],
        days_to_fetch=1,
        storage_base_path="example_data",
        circuit_breaker_config=cb_config,
        enable_validation=True,
        enable_events=True
    )
    
    async with DataIngestionService(config) as service:
        print(f"Fetching data with advanced configuration...")
        
        # Run the pipeline
        result = await service.run_ingestion_pipeline()
        
        print(f"\nAdvanced Results:")
        print(f"  Success: {result['success']}")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")
        
        # Show validation results
        if result['validation_result']:
            val_result = result['validation_result']
            print(f"  Validation: {val_result['status']} ({val_result['passed_checks']}/{val_result['total_checks']} checks)")
            
            if val_result['warnings']:
                print(f"  Warnings: {len(val_result['warnings'])}")
                for warning in val_result['warnings'][:3]:  # Show first 3
                    print(f"    - {warning}")
        
        # Show circuit breaker stats
        cb_stats = service.get_circuit_breaker_stats()
        print(f"  Circuit Breaker State: {cb_stats.get('state', 'unknown')}")
        
        # List available data
        available_data = service.list_available_data()
        print(f"\nAvailable Data:")
        for symbol, dates in available_data.items():
            print(f"  {symbol}: {len(dates)} days (latest: {dates[-1] if dates else 'none'})")


async def step_by_step_example():
    """Example showing individual pipeline steps."""
    print("\n=== Step-by-Step Pipeline Example ===")
    
    config = IngestionConfig(
        api_provider="yahoo",
        symbols=["AAPL"],
        days_to_fetch=1,
        storage_base_path="example_data"
    )
    
    async with DataIngestionService(config) as service:
        print("Step 1: Fetching data...")
        batch = await service.fetch_daily_data(["AAPL"], 1)
        print(f"  Fetched {len(batch.data)} records from {batch.source}")
        
        print("Step 2: Validating data...")
        validation_result = await service.validate_data_quality(batch)
        print(f"  Validation: {validation_result.status.value}")
        print(f"  Checks passed: {validation_result.passed_checks}/{validation_result.total_checks}")
        
        if validation_result.is_valid:
            print("Step 3: Storing data...")
            storage_result = await service.store_data(batch)
            print(f"  Storage success: {storage_result.success}")
            print(f"  Files written: {len(storage_result.files_written)}")
            
            if storage_result.success:
                print("Step 4: Publishing event...")
                event_result = await service.publish_data_event(batch, storage_result)
                print(f"  Event published: {event_result}")
        else:
            print("  Validation failed, skipping storage and events")
            for error in validation_result.errors:
                print(f"    Error: {error}")


async def error_handling_example():
    """Example demonstrating error handling."""
    print("\n=== Error Handling Example ===")
    
    # Configuration with invalid symbols to trigger errors
    config = IngestionConfig(
        api_provider="yahoo",
        symbols=["INVALID_SYMBOL_12345"],  # This should fail
        days_to_fetch=1,
        storage_base_path="example_data"
    )
    
    async with DataIngestionService(config) as service:
        print("Attempting to fetch data for invalid symbol...")
        
        result = await service.run_ingestion_pipeline()
        
        print(f"Results:")
        print(f"  Success: {result['success']}")
        print(f"  Records fetched: {result['records_fetched']}")
        
        if result['errors']:
            print(f"  Errors encountered:")
            for error in result['errors']:
                print(f"    - {error}")
        
        # Show circuit breaker stats after errors
        cb_stats = service.get_circuit_breaker_stats()
        if cb_stats:
            print(f"  Circuit breaker failures: {cb_stats.get('failure_count', 0)}")


async def main():
    """Run all examples."""
    try:
        await basic_ingestion_example()
        await advanced_ingestion_example()
        await step_by_step_example()
        await error_handling_example()
        
        print("\n=== All Examples Completed ===")
        print("Check the 'example_data' directory for stored data files.")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        logging.exception("Example execution failed")


if __name__ == "__main__":
    asyncio.run(main())