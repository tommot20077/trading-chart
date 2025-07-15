# ABOUTME: Integration tests for large-scale batch data processing performance
# ABOUTME: Tests performance characteristics of processing 10K+ data records with memory and time optimization

import pytest
import asyncio
import time
import psutil
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from core.interfaces.data.converter import AbstractDataConverter
from core.models.data.kline import Kline


@pytest.mark.integration
@pytest.mark.benchmark
@pytest.mark.asyncio
class TestBatchProcessingPerformance:
    """Integration tests for large-scale batch data processing performance"""

    async def test_large_dataset_kline_processing_performance(
        self, test_converter: AbstractDataConverter, mock_data_generator
    ):
        """Test processing performance with 10K+ K-line records"""
        # Arrange: Generate large dataset
        dataset_sizes = [1000, 5000, 10000, 20000]
        performance_results = []

        for size in dataset_sizes:
            print(f"\n=== Testing dataset size: {size} ===")

            # Generate test data
            large_dataset = mock_data_generator.generate_binance_kline_data("BTCUSDT", size)

            # Measure memory before processing
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Act: Process large dataset with timing
            start_time = time.perf_counter()

            converted_klines = test_converter.convert_multiple_klines(large_dataset, "BTCUSDT")

            end_time = time.perf_counter()
            processing_time = end_time - start_time

            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            # Calculate performance metrics
            records_per_second = size / processing_time if processing_time > 0 else 0
            memory_per_record = memory_used / size if size > 0 else 0

            performance_result = {
                "dataset_size": size,
                "processing_time": processing_time,
                "records_per_second": records_per_second,
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_used_mb": memory_used,
                "memory_per_record_kb": memory_per_record * 1024,
                "success_rate": len(converted_klines) / size,
            }

            performance_results.append(performance_result)

            print(f"Processing time: {processing_time:.3f}s")
            print(f"Records per second: {records_per_second:.1f}")
            print(f"Memory used: {memory_used:.2f}MB")
            print(f"Memory per record: {memory_per_record * 1024:.2f}KB")

            # Assert: Verify processing results
            assert len(converted_klines) == size, f"Should process all {size} records"
            assert all(isinstance(kline, Kline) for kline in converted_klines)

            # Performance assertions
            assert records_per_second >= 1000, f"Should process at least 1000 records/sec, got {records_per_second:.1f}"
            assert memory_per_record < 0.1, (
                f"Should use less than 100KB per record, got {memory_per_record * 1024:.2f}KB"
            )

        # Assert: Performance should scale reasonably
        print("\n=== Performance Summary ===")
        for result in performance_results:
            print(
                f"Size: {result['dataset_size']:5d} | "
                f"Time: {result['processing_time']:6.3f}s | "
                f"Rate: {result['records_per_second']:8.1f} rec/s | "
                f"Memory: {result['memory_used_mb']:6.2f}MB"
            )

        # Verify performance doesn't degrade significantly with size
        if len(performance_results) >= 2:
            first_rate = performance_results[0]["records_per_second"]
            last_rate = performance_results[-1]["records_per_second"]
            performance_ratio = last_rate / first_rate if first_rate > 0 else 0

            assert performance_ratio >= 0.5, (
                f"Performance should not degrade more than 50%, got {performance_ratio:.2f}"
            )

    async def test_concurrent_batch_processing_performance(
        self, test_converter: AbstractDataConverter, mock_data_generator
    ):
        """Test concurrent processing of multiple data batches"""
        # Arrange: Multiple datasets for concurrent processing
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
        records_per_symbol = 5000

        datasets = {
            symbol: mock_data_generator.generate_binance_kline_data(symbol, records_per_symbol) for symbol in symbols
        }

        # Test different concurrency levels
        concurrency_levels = [1, 2, 4]
        concurrency_results = []

        for max_workers in concurrency_levels:
            print(f"\n=== Testing concurrency level: {max_workers} ===")

            # Measure memory before processing
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024

            # Act: Process datasets concurrently
            start_time = time.perf_counter()

            async def process_symbol_data(symbol: str, data: List[Dict[str, Any]]):
                """Process data for a single symbol"""
                return symbol, test_converter.convert_multiple_klines(data, symbol)

            # Use ThreadPoolExecutor for I/O-bound tasks simulation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        executor,
                        lambda s=symbol, d=data: (s, test_converter.convert_multiple_klines(d, s)),
                        symbol,
                        data,
                    )
                    for symbol, data in datasets.items()
                ]

                results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            processing_time = end_time - start_time

            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            # Calculate metrics
            total_records = len(symbols) * records_per_symbol
            records_per_second = total_records / processing_time

            concurrency_result = {
                "max_workers": max_workers,
                "total_records": total_records,
                "processing_time": processing_time,
                "records_per_second": records_per_second,
                "memory_used_mb": memory_used,
                "symbols_processed": len(results),
            }

            concurrency_results.append(concurrency_result)

            print(f"Processing time: {processing_time:.3f}s")
            print(f"Records per second: {records_per_second:.1f}")
            print(f"Memory used: {memory_used:.2f}MB")

            # Assert: Verify processing results
            assert len(results) == len(symbols)
            for symbol, converted_klines in results:
                assert len(converted_klines) == records_per_symbol
                assert all(kline.symbol == symbol for kline in converted_klines)

        # Assert: Concurrency should not significantly degrade performance
        print("\n=== Concurrency Performance Summary ===")
        for result in concurrency_results:
            print(
                f"Workers: {result['max_workers']} | "
                f"Time: {result['processing_time']:6.3f}s | "
                f"Rate: {result['records_per_second']:8.1f} rec/s | "
                f"Memory: {result['memory_used_mb']:6.2f}MB"
            )

        # Verify concurrency benefits (realistic expectations for CPU-bound tasks)
        if len(concurrency_results) >= 2:
            sequential_time = concurrency_results[0]["processing_time"]
            concurrent_time = concurrency_results[-1]["processing_time"]
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0

            print(f"Speedup achieved: {speedup:.2f}x")

            # For CPU-bound tasks in single-core environments, concurrency may actually slow things down
            # This is expected behavior due to context switching overhead
            assert speedup >= 0.3, f"Concurrency should not severely degrade performance, got {speedup:.2f}x"

            # If we do get speedup, it's a bonus
            if speedup >= 1.1:
                print(f"[SUCCESS] Achieved {speedup:.2f}x speedup with concurrency")
            else:
                print(f"[INFO] CPU-bound task showed {speedup:.2f}x performance ratio (expected for single-core)")

    async def test_streaming_batch_processing_performance(
        self, test_converter: AbstractDataConverter, mock_data_generator
    ):
        """Test streaming processing for continuous data flow simulation"""
        # Arrange: Simulate streaming data processing
        total_records = 10000
        stream_batch_size = 100
        processing_interval = 0.01  # 10ms between batches

        # Generate data stream
        all_data = mock_data_generator.generate_binance_kline_data("BTCUSDT", total_records)

        # Measure memory before streaming
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        # Act: Process data in streaming fashion
        start_time = time.perf_counter()
        processed_count = 0
        all_results = []
        processing_times = []
        memory_snapshots = []

        for i in range(0, total_records, stream_batch_size):
            batch_start = time.perf_counter()

            # Get next batch
            batch = all_data[i : i + stream_batch_size]

            # Process batch
            batch_results = test_converter.convert_multiple_klines(batch, "BTCUSDT")
            all_results.extend(batch_results)
            processed_count += len(batch_results)

            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            processing_times.append(batch_time)

            # Monitor memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_snapshots.append(current_memory - memory_before)

            # Simulate streaming delay
            await asyncio.sleep(processing_interval)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate streaming metrics
        avg_batch_time = sum(processing_times) / len(processing_times)
        max_batch_time = max(processing_times)
        min_batch_time = min(processing_times)
        avg_memory = sum(memory_snapshots) / len(memory_snapshots)
        max_memory = max(memory_snapshots)

        streaming_metrics = {
            "total_records": total_records,
            "stream_batch_size": stream_batch_size,
            "total_time": total_time,
            "processing_time": sum(processing_times),
            "avg_batch_time": avg_batch_time,
            "max_batch_time": max_batch_time,
            "min_batch_time": min_batch_time,
            "batch_time_variance": max_batch_time - min_batch_time,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max_memory,
            "records_per_second": processed_count / sum(processing_times),
            "batches_processed": len(processing_times),
        }

        print("\n=== Streaming Processing Metrics ===")
        print(f"Total records: {streaming_metrics['total_records']}")
        print(f"Batch size: {streaming_metrics['stream_batch_size']}")
        print(f"Avg batch time: {streaming_metrics['avg_batch_time']:.4f}s")
        print(f"Max batch time: {streaming_metrics['max_batch_time']:.4f}s")
        print(f"Time variance: {streaming_metrics['batch_time_variance']:.4f}s")
        print(f"Avg memory: {streaming_metrics['avg_memory_mb']:.2f}MB")
        print(f"Max memory: {streaming_metrics['max_memory_mb']:.2f}MB")
        print(f"Processing rate: {streaming_metrics['records_per_second']:.1f} rec/s")

        # Assert: Verify streaming processing results
        assert processed_count == total_records
        assert len(all_results) == total_records
        assert all(isinstance(kline, Kline) for kline in all_results)

        # Streaming performance assertions
        assert avg_batch_time < 0.1, f"Average batch time should be under 100ms, got {avg_batch_time:.4f}s"
        assert streaming_metrics["batch_time_variance"] < 0.1, (
            f"Batch time variance should be under 100ms, got {streaming_metrics['batch_time_variance']:.4f}s"
        )
        assert max_memory < 50, f"Memory usage should stay under 50MB, got {max_memory:.2f}MB"
