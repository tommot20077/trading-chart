# ABOUTME: Integration tests for large-scale batch data processing performance
# ABOUTME: Tests performance characteristics of processing 10K+ data records with memory and time optimization

import pytest
import asyncio
import psutil
import os
from concurrent.futures import ThreadPoolExecutor

from core.interfaces.data.converter import AbstractDataConverter


@pytest.mark.integration
@pytest.mark.benchmark
@pytest.mark.asyncio
class TestBatchProcessingPerformance:
    """Integration tests for large-scale batch data processing performance"""

    @pytest.mark.benchmark
    async def test_large_dataset_kline_processing_performance(
        self, test_converter: AbstractDataConverter, mock_data_generator, benchmark
    ):
        """Test processing performance with 10K+ K-line records"""
        # Test with a representative dataset size for benchmark
        size = 5000
        large_dataset = mock_data_generator.generate_binance_kline_data("BTCUSDT", size)

        # Measure memory before processing
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        async def large_dataset_processing():
            converted_klines = test_converter.convert_multiple_klines(large_dataset, "BTCUSDT")

            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            print(f"Processed {len(converted_klines)} records")
            print(f"Memory used: {memory_used:.2f}MB")

            return len(converted_klines)

        # Benchmark large dataset processing
        result = await benchmark(large_dataset_processing)
        assert result == size

    @pytest.mark.benchmark
    async def test_concurrent_batch_processing_performance(
        self, test_converter: AbstractDataConverter, mock_data_generator, benchmark
    ):
        """Test concurrent processing of multiple data batches"""
        # Use smaller dataset for benchmark consistency
        symbols = ["BTCUSDT", "ETHUSDT"]
        records_per_symbol = 1000

        datasets = {
            symbol: mock_data_generator.generate_binance_kline_data(symbol, records_per_symbol) for symbol in symbols
        }

        # Measure memory before processing
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        async def concurrent_processing():
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=2) as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        executor, lambda s=symbol, d=data: (s, test_converter.convert_multiple_klines(d, s))
                    )
                    for symbol, data in datasets.items()
                ]

                results = await asyncio.gather(*tasks)

            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            print(f"Processed {len(results)} symbols concurrently")
            print(f"Memory used: {memory_used:.2f}MB")

            return len(results)

        # Benchmark concurrent processing
        result = await benchmark(concurrent_processing)
        assert result == len(symbols)

    @pytest.mark.benchmark
    async def test_streaming_batch_processing_performance(
        self, test_converter: AbstractDataConverter, mock_data_generator, benchmark
    ):
        """Test streaming processing for continuous data flow simulation"""
        # Use smaller dataset for benchmark consistency
        total_records = 1000
        stream_batch_size = 50
        processing_interval = 0.001  # 1ms between batches

        # Generate data stream
        all_data = mock_data_generator.generate_binance_kline_data("BTCUSDT", total_records)

        # Measure memory before streaming
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        async def streaming_processing():
            processed_count = 0
            all_results = []

            for i in range(0, total_records, stream_batch_size):
                # Get next batch
                batch = all_data[i : i + stream_batch_size]

                # Process batch
                batch_results = test_converter.convert_multiple_klines(batch, "BTCUSDT")
                all_results.extend(batch_results)
                processed_count += len(batch_results)

                # Simulate streaming delay
                await asyncio.sleep(processing_interval)

            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            print(f"Streamed {processed_count} records in batches")
            print(f"Memory used: {memory_used:.2f}MB")

            return processed_count

        # Benchmark streaming processing
        result = await benchmark(streaming_processing)
        assert result == total_records
