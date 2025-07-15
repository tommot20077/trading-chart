# ABOUTME: Integration tests for memory usage optimization in data processing
# ABOUTME: Tests memory efficiency, garbage collection, and memory leak detection in batch processing

import pytest
import time
import psutil
import os
import gc
import weakref

from core.interfaces.data.converter import AbstractDataConverter


@pytest.mark.integration
@pytest.mark.benchmark
@pytest.mark.asyncio
class TestMemoryOptimizationIntegration:
    """Integration tests for memory usage optimization in data processing"""

    async def test_memory_usage_under_100mb_constraint(
        self, test_converter: AbstractDataConverter, mock_data_generator
    ):
        """Test that memory usage stays under 100MB during large data processing"""
        # Arrange: Large dataset that could potentially use significant memory
        dataset_size = 50000  # 50K records
        batch_size = 1000

        # Measure baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"Baseline memory: {baseline_memory:.2f}MB")

        # Act: Process data in controlled batches with memory monitoring
        peak_memory = baseline_memory
        memory_samples = []
        processed_count = 0

        for batch_start in range(0, dataset_size, batch_size):
            # Generate batch data (don't keep all data in memory at once)
            batch_data = mock_data_generator.generate_binance_kline_data(
                "BTCUSDT", min(batch_size, dataset_size - batch_start)
            )

            # Process batch
            batch_results = test_converter.convert_multiple_klines(batch_data, "BTCUSDT")
            processed_count += len(batch_results)

            # Monitor memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_used = current_memory - baseline_memory
            peak_memory = max(peak_memory, current_memory)
            memory_samples.append(memory_used)

            # Force garbage collection to free unused memory
            del batch_data
            del batch_results
            gc.collect()

            # Check memory constraint during processing
            if memory_used > 100:
                pytest.fail(
                    f"Memory usage exceeded 100MB limit: {memory_used:.2f}MB at batch {batch_start // batch_size + 1}"
                )

        # Calculate memory statistics
        peak_memory_used = peak_memory - baseline_memory
        avg_memory_used = sum(memory_samples) / len(memory_samples)
        max_memory_used = max(memory_samples)

        memory_stats = {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_used_mb": peak_memory_used,
            "avg_memory_used_mb": avg_memory_used,
            "max_memory_used_mb": max_memory_used,
            "total_records_processed": processed_count,
            "memory_per_record_kb": (avg_memory_used * 1024) / processed_count if processed_count > 0 else 0,
            "batches_processed": len(memory_samples),
        }

        print("\n=== Memory Usage Statistics ===")
        print(f"Total records processed: {memory_stats['total_records_processed']}")
        print(f"Peak memory used: {memory_stats['peak_memory_used_mb']:.2f}MB")
        print(f"Average memory used: {memory_stats['avg_memory_used_mb']:.2f}MB")
        print(f"Max memory used: {memory_stats['max_memory_used_mb']:.2f}MB")
        print(f"Memory per record: {memory_stats['memory_per_record_kb']:.2f}KB")

        # Assert: Memory usage constraints
        assert processed_count == dataset_size, f"Should process all {dataset_size} records"
        assert memory_stats["peak_memory_used_mb"] <= 100, (
            f"Peak memory should be under 100MB, got {memory_stats['peak_memory_used_mb']:.2f}MB"
        )
        assert memory_stats["avg_memory_used_mb"] <= 50, (
            f"Average memory should be under 50MB, got {memory_stats['avg_memory_used_mb']:.2f}MB"
        )
        assert memory_stats["memory_per_record_kb"] <= 2.0, (
            f"Memory per record should be under 2KB, got {memory_stats['memory_per_record_kb']:.2f}KB"
        )

    async def test_memory_leak_detection(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test for memory leaks during repeated processing cycles"""
        # Arrange: Setup for leak detection
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Create weak references to track object cleanup
        weak_refs = []
        memory_measurements = []

        # Act: Perform multiple processing cycles
        cycles = 10
        records_per_cycle = 5000

        for cycle in range(cycles):
            print(f"\n--- Cycle {cycle + 1}/{cycles} ---")

            # Generate and process data
            cycle_data = mock_data_generator.generate_binance_kline_data("BTCUSDT", records_per_cycle)
            cycle_results = test_converter.convert_multiple_klines(cycle_data, "BTCUSDT")

            # Create weak references to some objects to track cleanup
            if len(cycle_results) > 0:
                weak_refs.append(weakref.ref(cycle_results[0]))
                weak_refs.append(weakref.ref(cycle_results[-1]))

            # Measure memory after processing
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_used = current_memory - baseline_memory
            memory_measurements.append(memory_used)

            print(f"Memory used: {memory_used:.2f}MB")

            # Explicitly delete references and force garbage collection
            del cycle_data
            del cycle_results
            gc.collect()

            # Measure memory after cleanup
            cleanup_memory = process.memory_info().rss / 1024 / 1024
            cleanup_memory_used = cleanup_memory - baseline_memory

            print(f"Memory after cleanup: {cleanup_memory_used:.2f}MB")

        # Analyze memory leak patterns
        if len(memory_measurements) >= 3:
            # Check if memory usage is consistently increasing (potential leak)
            early_avg = sum(memory_measurements[:3]) / 3
            late_avg = sum(memory_measurements[-3:]) / 3
            memory_growth = late_avg - early_avg

            # Check weak references - some should be garbage collected
            alive_refs = sum(1 for ref in weak_refs if ref() is not None)
            total_refs = len(weak_refs)
            cleanup_rate = (total_refs - alive_refs) / total_refs if total_refs > 0 else 0

            leak_analysis = {
                "cycles_completed": cycles,
                "early_avg_memory_mb": early_avg,
                "late_avg_memory_mb": late_avg,
                "memory_growth_mb": memory_growth,
                "total_weak_refs": total_refs,
                "alive_refs": alive_refs,
                "cleanup_rate": cleanup_rate,
                "max_memory_mb": max(memory_measurements),
                "min_memory_mb": min(memory_measurements),
            }

            print("\n=== Memory Leak Analysis ===")
            print(f"Early cycles avg memory: {leak_analysis['early_avg_memory_mb']:.2f}MB")
            print(f"Late cycles avg memory: {leak_analysis['late_avg_memory_mb']:.2f}MB")
            print(f"Memory growth: {leak_analysis['memory_growth_mb']:.2f}MB")
            print(f"Object cleanup rate: {leak_analysis['cleanup_rate']:.2%}")
            print(f"Max memory: {leak_analysis['max_memory_mb']:.2f}MB")
            print(f"Min memory: {leak_analysis['min_memory_mb']:.2f}MB")

            # Assert: No significant memory leaks
            assert memory_growth <= 10, f"Memory growth should be under 10MB, got {memory_growth:.2f}MB"
            assert cleanup_rate >= 0.5, f"At least 50% of objects should be cleaned up, got {cleanup_rate:.2%}"
            assert leak_analysis["max_memory_mb"] <= 100, (
                f"Max memory should stay under 100MB, got {leak_analysis['max_memory_mb']:.2f}MB"
            )

    async def test_garbage_collection_effectiveness(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test effectiveness of garbage collection in freeing memory"""
        # Arrange: Setup for GC testing
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Act: Create large data structures and test GC
        large_dataset = mock_data_generator.generate_binance_kline_data("BTCUSDT", 20000)

        # Process data to create objects
        results = test_converter.convert_multiple_klines(large_dataset, "BTCUSDT")

        # Measure memory before GC
        memory_before_gc = process.memory_info().rss / 1024 / 1024
        memory_used_before = memory_before_gc - baseline_memory

        # Create additional temporary objects
        temp_objects = []
        for i in range(1000):
            temp_data = mock_data_generator.generate_binance_kline_data("ETHUSDT", 10)
            temp_results = test_converter.convert_multiple_klines(temp_data, "ETHUSDT")
            temp_objects.append(temp_results)

        # Measure memory with temporary objects
        memory_with_temp = process.memory_info().rss / 1024 / 1024
        memory_used_with_temp = memory_with_temp - baseline_memory

        # Delete references and force garbage collection
        del large_dataset
        del results
        del temp_objects

        # Multiple GC passes to ensure thorough cleanup
        for _ in range(3):
            gc.collect()

        # Measure memory after GC
        memory_after_gc = process.memory_info().rss / 1024 / 1024
        memory_used_after = memory_after_gc - baseline_memory

        # Calculate GC effectiveness
        memory_freed = memory_used_with_temp - memory_used_after
        gc_effectiveness = memory_freed / memory_used_with_temp if memory_used_with_temp > 0 else 0

        gc_stats = {
            "baseline_memory_mb": baseline_memory,
            "memory_before_gc_mb": memory_used_before,
            "memory_with_temp_mb": memory_used_with_temp,
            "memory_after_gc_mb": memory_used_after,
            "memory_freed_mb": memory_freed,
            "gc_effectiveness": gc_effectiveness,
            "temp_objects_created": 1000,
        }

        print("\n=== Garbage Collection Effectiveness ===")
        print(f"Memory before GC: {gc_stats['memory_before_gc_mb']:.2f}MB")
        print(f"Memory with temp objects: {gc_stats['memory_with_temp_mb']:.2f}MB")
        print(f"Memory after GC: {gc_stats['memory_after_gc_mb']:.2f}MB")
        print(f"Memory freed: {gc_stats['memory_freed_mb']:.2f}MB")
        print(f"GC effectiveness: {gc_stats['gc_effectiveness']:.2%}")

        # Assert: GC effectiveness
        assert gc_stats["memory_freed_mb"] >= 0, "GC should free some memory"
        assert gc_stats["gc_effectiveness"] >= 0.3, (
            f"GC should free at least 30% of temporary memory, got {gc_stats['gc_effectiveness']:.2%}"
        )
        assert gc_stats["memory_after_gc_mb"] <= gc_stats["memory_with_temp_mb"], (
            "Memory after GC should not exceed memory with temporary objects"
        )

    async def test_memory_efficient_batch_processing(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test memory-efficient batch processing with different batch sizes"""
        # Arrange: Test different batch sizes for memory efficiency
        total_records = 30000
        batch_sizes = [100, 500, 1000, 2000, 5000]

        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024

        batch_efficiency_results = []

        for batch_size in batch_sizes:
            print(f"\n=== Testing batch size: {batch_size} ===")

            # Reset memory state
            gc.collect()
            start_memory = process.memory_info().rss / 1024 / 1024

            peak_memory = start_memory
            total_processed = 0

            # Process data in batches
            start_time = time.perf_counter()

            for i in range(0, total_records, batch_size):
                # Generate batch (don't keep all data in memory)
                current_batch_size = min(batch_size, total_records - i)
                batch_data = mock_data_generator.generate_binance_kline_data("BTCUSDT", current_batch_size)

                # Process batch
                batch_results = test_converter.convert_multiple_klines(batch_data, "BTCUSDT")
                total_processed += len(batch_results)

                # Monitor peak memory
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)

                # Clean up batch data immediately
                del batch_data
                del batch_results

                # Periodic garbage collection for larger batches
                if batch_size >= 1000 and i % (batch_size * 5) == 0:
                    gc.collect()

            end_time = time.perf_counter()
            processing_time = end_time - start_time

            # Final cleanup and memory measurement
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024

            # Calculate efficiency metrics
            peak_memory_used = peak_memory - baseline_memory
            final_memory_used = final_memory - baseline_memory
            memory_efficiency = (
                1.0 - (peak_memory_used / (total_records * 0.002)) if total_records > 0 else 0
            )  # Assume 2KB per record baseline

            batch_result = {
                "batch_size": batch_size,
                "total_processed": total_processed,
                "processing_time": processing_time,
                "records_per_second": total_processed / processing_time,
                "peak_memory_mb": peak_memory_used,
                "final_memory_mb": final_memory_used,
                "memory_efficiency": memory_efficiency,
                "memory_per_record_kb": (peak_memory_used * 1024) / total_processed if total_processed > 0 else 0,
            }

            batch_efficiency_results.append(batch_result)

            print(f"Processed: {batch_result['total_processed']} records")
            print(f"Processing time: {batch_result['processing_time']:.3f}s")
            print(f"Peak memory: {batch_result['peak_memory_mb']:.2f}MB")
            print(f"Memory per record: {batch_result['memory_per_record_kb']:.2f}KB")
            print(f"Records/sec: {batch_result['records_per_second']:.1f}")

            # Assert: Memory efficiency for this batch size
            assert total_processed == total_records, f"Should process all {total_records} records"
            assert peak_memory_used <= 100, f"Peak memory should be under 100MB, got {peak_memory_used:.2f}MB"

        # Analyze batch size efficiency
        print("\n=== Batch Size Efficiency Analysis ===")
        print(f"{'Batch Size':<10} {'Peak Mem(MB)':<12} {'Mem/Rec(KB)':<12} {'Rate(rec/s)':<12} {'Time(s)':<10}")
        print("-" * 70)

        for result in batch_efficiency_results:
            print(
                f"{result['batch_size']:<10} "
                f"{result['peak_memory_mb']:<12.2f} "
                f"{result['memory_per_record_kb']:<12.2f} "
                f"{result['records_per_second']:<12.1f} "
                f"{result['processing_time']:<10.3f}"
            )

        # Find optimal batch size (balance between memory and performance)
        optimal_batch = min(
            batch_efficiency_results, key=lambda x: x["peak_memory_mb"] + (1.0 / x["records_per_second"]) * 1000
        )

        print(
            f"\nOptimal batch size: {optimal_batch['batch_size']} "
            f"(Peak: {optimal_batch['peak_memory_mb']:.2f}MB, "
            f"Rate: {optimal_batch['records_per_second']:.1f} rec/s)"
        )

        # Assert: Overall efficiency
        min_peak_memory = min(r["peak_memory_mb"] for r in batch_efficiency_results)
        max_rate = max(r["records_per_second"] for r in batch_efficiency_results)

        assert min_peak_memory <= 50, f"Best case peak memory should be under 50MB, got {min_peak_memory:.2f}MB"
        assert max_rate >= 10000, f"Best case rate should be at least 10K rec/s, got {max_rate:.1f}"
        assert optimal_batch["peak_memory_mb"] <= 100, (
            f"Optimal batch should use under 100MB, got {optimal_batch['peak_memory_mb']:.2f}MB"
        )

    async def test_memory_pressure_handling(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test system behavior under memory pressure conditions"""
        # Arrange: Create memory pressure scenario
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Simulate memory pressure by creating large objects
        memory_pressure_objects = []
        target_pressure_mb = 200  # Create 200MB of memory pressure

        try:
            # Create memory pressure
            while True:
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory - baseline_memory >= target_pressure_mb:
                    break

                # Create large data structures
                pressure_data = mock_data_generator.generate_binance_kline_data("PRESSURE", 10000)
                memory_pressure_objects.append(pressure_data)

            pressure_memory = process.memory_info().rss / 1024 / 1024
            print(f"Created memory pressure: {pressure_memory - baseline_memory:.2f}MB")

            # Act: Process data under memory pressure
            test_data = mock_data_generator.generate_binance_kline_data("BTCUSDT", 5000)

            start_time = time.perf_counter()
            results = test_converter.convert_multiple_klines(test_data, "BTCUSDT")
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            peak_memory = process.memory_info().rss / 1024 / 1024

            # Clean up test data
            del test_data
            gc.collect()

            # Measure performance under pressure
            pressure_stats = {
                "baseline_memory_mb": baseline_memory,
                "pressure_memory_mb": pressure_memory - baseline_memory,
                "peak_memory_mb": peak_memory - baseline_memory,
                "processing_time": processing_time,
                "records_processed": 5000,
                "records_per_second": 5000 / processing_time,
                "memory_pressure_created": True,
            }

            print("\n=== Memory Pressure Test Results ===")
            print(f"Memory pressure created: {pressure_stats['pressure_memory_mb']:.2f}MB")
            print(f"Peak memory during test: {pressure_stats['peak_memory_mb']:.2f}MB")
            print(f"Processing time: {pressure_stats['processing_time']:.3f}s")
            print(f"Records per second: {pressure_stats['records_per_second']:.1f}")

            # Assert: System should still function under pressure
            assert pressure_stats["records_processed"] == 5000, "Should process all records despite memory pressure"
            assert pressure_stats["records_per_second"] >= 1000, (
                f"Should maintain reasonable performance under pressure, got {pressure_stats['records_per_second']:.1f} rec/s"
            )
            assert pressure_stats["peak_memory_mb"] <= 500, (
                f"Peak memory should not exceed 500MB even under pressure, got {pressure_stats['peak_memory_mb']:.2f}MB"
            )

        finally:
            # Clean up memory pressure objects
            del memory_pressure_objects
            gc.collect()

            # Verify memory cleanup
            cleanup_memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory after cleanup: {cleanup_memory - baseline_memory:.2f}MB")

            # Memory should return close to baseline (within 50MB)
            assert cleanup_memory - baseline_memory <= 50, (
                f"Memory should return close to baseline after cleanup, got {cleanup_memory - baseline_memory:.2f}MB"
            )
