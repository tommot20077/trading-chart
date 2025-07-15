# ABOUTME: Integration tests for parallel data processing with multi-threading and multi-processing
# ABOUTME: Tests concurrent processing, thread safety, and parallel execution patterns

import pytest
import asyncio
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from core.interfaces.data.converter import AbstractDataConverter
from core.models.data.kline import Kline


@pytest.mark.integration
@pytest.mark.benchmark
@pytest.mark.asyncio
class TestParallelProcessingIntegration:
    """Integration tests for parallel data processing"""

    async def test_multi_threaded_data_processing(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test multi-threaded processing of data batches"""
        # Arrange: Prepare data for parallel processing
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
        records_per_symbol = 5000

        datasets = {
            symbol: mock_data_generator.generate_binance_kline_data(symbol, records_per_symbol) for symbol in symbols
        }

        # Test different thread counts
        thread_counts = [1, 2, 4, 8]
        threading_results = []

        for max_workers in thread_counts:
            print(f"\n=== Testing with {max_workers} threads ===")

            # Measure memory and time
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024

            start_time = time.perf_counter()

            # Act: Process data using ThreadPoolExecutor
            def process_symbol_data(symbol_data_pair):
                symbol, data = symbol_data_pair
                return symbol, test_converter.convert_multiple_klines(data, symbol)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(process_symbol_data, (symbol, data)): symbol for symbol, data in datasets.items()
                }

                # Collect results
                results = {}
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        symbol_name, converted_data = future.result()
                        results[symbol_name] = converted_data
                    except Exception as exc:
                        pytest.fail(f"Symbol {symbol} generated an exception: {exc}")

            end_time = time.perf_counter()
            processing_time = end_time - start_time

            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            # Calculate metrics
            total_records = len(symbols) * records_per_symbol
            records_per_second = total_records / processing_time

            threading_result = {
                "max_workers": max_workers,
                "total_records": total_records,
                "processing_time": processing_time,
                "records_per_second": records_per_second,
                "memory_used_mb": memory_used,
                "symbols_processed": len(results),
            }

            threading_results.append(threading_result)

            print(f"Processing time: {processing_time:.3f}s")
            print(f"Records per second: {records_per_second:.1f}")
            print(f"Memory used: {memory_used:.2f}MB")

            # Assert: Verify processing results
            assert len(results) == len(symbols), f"Should process all {len(symbols)} symbols"
            for symbol, converted_klines in results.items():
                assert len(converted_klines) == records_per_symbol, (
                    f"Symbol {symbol} should have {records_per_symbol} records"
                )
                assert all(isinstance(kline, Kline) for kline in converted_klines), (
                    f"All results for {symbol} should be Kline objects"
                )
                assert all(kline.symbol == symbol for kline in converted_klines), (
                    f"All klines should have correct symbol {symbol}"
                )

        # Analyze threading performance
        print("\n=== Multi-Threading Performance Analysis ===")
        print(f"{'Threads':<8} {'Time(s)':<8} {'Rate(rec/s)':<12} {'Memory(MB)':<12} {'Speedup':<8}")
        print("-" * 60)

        baseline_time = threading_results[0]["processing_time"]
        for result in threading_results:
            speedup = baseline_time / result["processing_time"]
            print(
                f"{result['max_workers']:<8} "
                f"{result['processing_time']:<8.3f} "
                f"{result['records_per_second']:<12.1f} "
                f"{result['memory_used_mb']:<12.2f} "
                f"{speedup:<8.2f}"
            )

        # Assert: Threading should provide some benefit or at least not degrade significantly
        best_performance = max(threading_results, key=lambda x: x["records_per_second"])
        worst_performance = min(threading_results, key=lambda x: x["records_per_second"])

        performance_ratio = best_performance["records_per_second"] / worst_performance["records_per_second"]

        assert performance_ratio <= 3.0, (
            f"Performance variation should be reasonable, got {performance_ratio:.2f}x difference"
        )
        assert best_performance["records_per_second"] >= 10000, (
            f"Best threading performance should be at least 10K rec/s, got {best_performance['records_per_second']:.1f}"
        )

    async def test_thread_safety_verification(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test thread safety of data converter under concurrent access"""
        # Arrange: Setup for thread safety testing
        shared_results = {}
        results_lock = Lock()
        error_count = 0
        error_lock = Lock()

        # Create test data
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        records_per_symbol = 1000

        def worker_function(worker_id: int, symbol: str):
            """Worker function for thread safety testing"""
            nonlocal error_count

            try:
                # Generate data for this worker
                worker_data = mock_data_generator.generate_binance_kline_data(symbol, records_per_symbol)

                # Process data using shared converter
                worker_results = test_converter.convert_multiple_klines(worker_data, symbol)

                # Store results safely
                with results_lock:
                    if symbol not in shared_results:
                        shared_results[symbol] = []
                    shared_results[symbol].extend(worker_results)

                # Verify results integrity
                assert len(worker_results) == records_per_symbol
                assert all(isinstance(kline, Kline) for kline in worker_results)
                assert all(kline.symbol == symbol for kline in worker_results)

            except Exception as e:
                with error_lock:
                    error_count += 1
                print(f"Worker {worker_id} error: {e}")

        # Act: Run multiple workers concurrently
        workers_per_symbol = 5
        total_workers = len(test_symbols) * workers_per_symbol

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = []
            worker_id = 0

            for symbol in test_symbols:
                for _ in range(workers_per_symbol):
                    future = executor.submit(worker_function, worker_id, symbol)
                    futures.append(future)
                    worker_id += 1

            # Wait for all workers to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    print(f"Future exception: {e}")

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Analyze thread safety results
        total_records_expected = len(test_symbols) * workers_per_symbol * records_per_symbol
        total_records_actual = sum(len(results) for results in shared_results.values())

        thread_safety_stats = {
            "total_workers": total_workers,
            "workers_per_symbol": workers_per_symbol,
            "symbols_tested": len(test_symbols),
            "processing_time": processing_time,
            "total_records_expected": total_records_expected,
            "total_records_actual": total_records_actual,
            "error_count": error_count,
            "success_rate": (total_workers - error_count) / total_workers,
            "records_per_second": total_records_actual / processing_time,
        }

        print("\n=== Thread Safety Test Results ===")
        print(f"Total workers: {thread_safety_stats['total_workers']}")
        print(f"Processing time: {thread_safety_stats['processing_time']:.3f}s")
        print(f"Records expected: {thread_safety_stats['total_records_expected']}")
        print(f"Records actual: {thread_safety_stats['total_records_actual']}")
        print(f"Error count: {thread_safety_stats['error_count']}")
        print(f"Success rate: {thread_safety_stats['success_rate']:.2%}")
        print(f"Records per second: {thread_safety_stats['records_per_second']:.1f}")

        # Assert: Thread safety verification
        assert error_count == 0, f"Should have no errors in thread safety test, got {error_count}"
        assert total_records_actual == total_records_expected, (
            f"Should process all expected records, got {total_records_actual}/{total_records_expected}"
        )
        assert thread_safety_stats["success_rate"] == 1.0, (
            f"All workers should succeed, got {thread_safety_stats['success_rate']:.2%}"
        )

        # Verify data integrity per symbol
        for symbol in test_symbols:
            symbol_results = shared_results[symbol]
            expected_count = workers_per_symbol * records_per_symbol
            assert len(symbol_results) == expected_count, (
                f"Symbol {symbol} should have {expected_count} records, got {len(symbol_results)}"
            )

    async def test_async_parallel_processing(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test asynchronous parallel processing patterns"""
        # Arrange: Setup for async parallel processing
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
        records_per_symbol = 3000

        async def process_symbol_async(symbol: str, semaphore: asyncio.Semaphore):
            """Async worker function with semaphore for concurrency control"""
            async with semaphore:
                # Simulate async data fetching delay
                await asyncio.sleep(0.01)

                # Generate and process data
                data = mock_data_generator.generate_binance_kline_data(symbol, records_per_symbol)

                # Process in executor to avoid blocking
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, test_converter.convert_multiple_klines, data, symbol)

                return symbol, results

        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        async_results = []

        for max_concurrent in concurrency_levels:
            print(f"\n=== Testing async concurrency: {max_concurrent} ===")

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)

            start_time = time.perf_counter()

            # Act: Process all symbols concurrently
            tasks = [process_symbol_async(symbol, semaphore) for symbol in symbols]
            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            processing_time = end_time - start_time

            # Calculate metrics
            total_records = len(symbols) * records_per_symbol
            records_per_second = total_records / processing_time

            async_result = {
                "max_concurrent": max_concurrent,
                "total_records": total_records,
                "processing_time": processing_time,
                "records_per_second": records_per_second,
                "symbols_processed": len(results),
            }

            async_results.append(async_result)

            print(f"Processing time: {processing_time:.3f}s")
            print(f"Records per second: {records_per_second:.1f}")

            # Assert: Verify async processing results
            assert len(results) == len(symbols), f"Should process all {len(symbols)} symbols"
            for symbol_name, converted_klines in results:
                assert len(converted_klines) == records_per_symbol, (
                    f"Symbol {symbol_name} should have {records_per_symbol} records"
                )
                assert all(isinstance(kline, Kline) for kline in converted_klines), (
                    f"All results for {symbol_name} should be Kline objects"
                )

        # Analyze async performance
        print("\n=== Async Parallel Processing Analysis ===")
        print(f"{'Concurrency':<12} {'Time(s)':<8} {'Rate(rec/s)':<12} {'Speedup':<8}")
        print("-" * 50)

        baseline_time = async_results[0]["processing_time"]
        for result in async_results:
            speedup = baseline_time / result["processing_time"]
            print(
                f"{result['max_concurrent']:<12} "
                f"{result['processing_time']:<8.3f} "
                f"{result['records_per_second']:<12.1f} "
                f"{speedup:<8.2f}"
            )

        # Assert: Async processing should be efficient
        best_async = max(async_results, key=lambda x: x["records_per_second"])
        assert best_async["records_per_second"] >= 5000, (
            f"Best async performance should be at least 5K rec/s, got {best_async['records_per_second']:.1f}"
        )

    async def test_parallel_processing_with_error_handling(
        self, test_converter: AbstractDataConverter, mock_data_generator
    ):
        """Test parallel processing with error injection and recovery"""
        # Arrange: Setup for error handling testing
        symbols = ["BTCUSDT", "ETHUSDT", "ERROR_SYMBOL", "BNBUSDT", "ANOTHER_ERROR"]
        records_per_symbol = 2000

        def process_with_potential_errors(symbol: str):
            """Worker function that may encounter errors"""
            try:
                if "ERROR" in symbol:
                    # Simulate processing error
                    raise ValueError(f"Simulated error for {symbol}")

                # Normal processing
                data = mock_data_generator.generate_binance_kline_data(symbol, records_per_symbol)
                results = test_converter.convert_multiple_klines(data, symbol)
                return {"symbol": symbol, "results": results, "status": "success", "error": None}

            except Exception as e:
                return {"symbol": symbol, "results": [], "status": "error", "error": str(e)}

        # Act: Process with error handling
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_with_potential_errors, symbol) for symbol in symbols]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Analyze error handling results
        successful_results = [r for r in results if r["status"] == "success"]
        error_results = [r for r in results if r["status"] == "error"]

        total_successful_records = sum(len(r["results"]) for r in successful_results)

        error_handling_stats = {
            "total_symbols": len(symbols),
            "successful_symbols": len(successful_results),
            "error_symbols": len(error_results),
            "total_successful_records": total_successful_records,
            "processing_time": processing_time,
            "success_rate": len(successful_results) / len(symbols),
            "records_per_second": total_successful_records / processing_time if processing_time > 0 else 0,
        }

        print("\n=== Parallel Error Handling Results ===")
        print(f"Total symbols: {error_handling_stats['total_symbols']}")
        print(f"Successful: {error_handling_stats['successful_symbols']}")
        print(f"Errors: {error_handling_stats['error_symbols']}")
        print(f"Success rate: {error_handling_stats['success_rate']:.2%}")
        print(f"Successful records: {error_handling_stats['total_successful_records']}")
        print(f"Records per second: {error_handling_stats['records_per_second']:.1f}")

        # Print error details
        for error_result in error_results:
            print(f"Error in {error_result['symbol']}: {error_result['error']}")

        # Assert: Error handling verification
        assert len(successful_results) == 3, "Should have 3 successful symbols"
        assert len(error_results) == 2, "Should have 2 error symbols"
        assert error_handling_stats["success_rate"] == 0.6, "Success rate should be 60%"
        assert total_successful_records == 3 * records_per_symbol, (
            f"Should have processed {3 * records_per_symbol} records successfully"
        )

        # Verify successful results
        for result in successful_results:
            assert len(result["results"]) == records_per_symbol
            assert all(isinstance(kline, Kline) for kline in result["results"])

        # Verify error results
        for result in error_results:
            assert "ERROR" in result["symbol"]
            assert len(result["results"]) == 0
            assert result["error"] is not None

    async def test_resource_contention_handling(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test handling of resource contention in parallel processing"""
        # Arrange: Setup for resource contention testing
        num_workers = 8
        records_per_worker = 2000

        # Shared resource simulation (using a lock to simulate contention)
        shared_counter = {"value": 0}
        contention_lock = Lock()
        processing_times = []

        def worker_with_contention(worker_id: int):
            """Worker that simulates resource contention"""
            start_time = time.perf_counter()

            # Generate data
            data = mock_data_generator.generate_binance_kline_data(f"SYMBOL{worker_id}", records_per_worker)

            # Process data
            results = test_converter.convert_multiple_klines(data, f"SYMBOL{worker_id}")

            # Simulate shared resource access (brief contention)
            with contention_lock:
                shared_counter["value"] += len(results)
                time.sleep(0.001)  # Simulate brief resource access

            end_time = time.perf_counter()
            worker_time = end_time - start_time

            return {
                "worker_id": worker_id,
                "processing_time": worker_time,
                "records_processed": len(results),
                "results": results,
            }

        # Act: Run workers with resource contention
        overall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_with_contention, i) for i in range(num_workers)]
            worker_results = [future.result() for future in as_completed(futures)]

        overall_end = time.perf_counter()
        total_time = overall_end - overall_start

        # Analyze contention handling
        total_records = sum(r["records_processed"] for r in worker_results)
        avg_worker_time = sum(r["processing_time"] for r in worker_results) / len(worker_results)
        max_worker_time = max(r["processing_time"] for r in worker_results)
        min_worker_time = min(r["processing_time"] for r in worker_results)

        contention_stats = {
            "num_workers": num_workers,
            "total_time": total_time,
            "total_records": total_records,
            "avg_worker_time": avg_worker_time,
            "max_worker_time": max_worker_time,
            "min_worker_time": min_worker_time,
            "time_variance": max_worker_time - min_worker_time,
            "shared_counter_value": shared_counter["value"],
            "records_per_second": total_records / total_time,
            "efficiency": (avg_worker_time * num_workers) / total_time,
        }

        print("\n=== Resource Contention Analysis ===")
        print(f"Workers: {contention_stats['num_workers']}")
        print(f"Total time: {contention_stats['total_time']:.3f}s")
        print(f"Total records: {contention_stats['total_records']}")
        print(f"Avg worker time: {contention_stats['avg_worker_time']:.3f}s")
        print(f"Max worker time: {contention_stats['max_worker_time']:.3f}s")
        print(f"Min worker time: {contention_stats['min_worker_time']:.3f}s")
        print(f"Time variance: {contention_stats['time_variance']:.3f}s")
        print(f"Shared counter: {contention_stats['shared_counter_value']}")
        print(f"Records/sec: {contention_stats['records_per_second']:.1f}")
        print(f"Efficiency: {contention_stats['efficiency']:.2f}")

        # Assert: Resource contention handling
        assert contention_stats["shared_counter_value"] == total_records, (
            "Shared counter should match total records processed"
        )
        assert contention_stats["time_variance"] <= 1.0, (
            f"Worker time variance should be reasonable, got {contention_stats['time_variance']:.3f}s"
        )
        assert contention_stats["efficiency"] >= 0.5, (
            f"Parallel efficiency should be at least 50%, got {contention_stats['efficiency']:.2f}"
        )
        assert contention_stats["records_per_second"] >= 5000, (
            f"Should maintain good throughput despite contention, got {contention_stats['records_per_second']:.1f}"
        )

        # Verify all workers completed successfully
        assert len(worker_results) == num_workers, f"All {num_workers} workers should complete"
        for result in worker_results:
            assert result["records_processed"] == records_per_worker, (
                f"Worker {result['worker_id']} should process {records_per_worker} records"
            )
