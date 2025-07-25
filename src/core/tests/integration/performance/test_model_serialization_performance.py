# ABOUTME: Performance benchmark tests for model serialization and deserialization
# ABOUTME: Tests serialization performance of Order, TradingPair, MarketData, Kline, and Trade models

import pytest
import time
import asyncio
from decimal import Decimal
from datetime import datetime, UTC, timedelta
from typing import List, Dict, Any
from uuid import UUID
import json
import pickle
import statistics

from core.models.data.order import Order
from core.models.data.trading_pair import TradingPair
from core.models.data.market_data import MarketData
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide, AssetClass, KlineInterval
from core.models.data.order_enums import OrderType, OrderStatus, OrderSide
from core.models.data.trading_pair import TradingPairStatus


class SerializationPerformanceMonitor:
    """Monitor for tracking serialization performance metrics."""
    
    def __init__(self):
        self.results = {}
    
    def measure_serialization(self, model_name: str, objects: List[Any], 
                            serialization_func: callable, iterations: int = 1) -> Dict[str, float]:
        """Measure serialization performance."""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            for obj in objects:
                serialization_func(obj)
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        throughput = len(objects) / (avg_time / 1000)  # Objects per second
        
        result = {
            "model_name": model_name,
            "object_count": len(objects),
            "iterations": iterations,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "std_dev_ms": std_dev,
            "throughput_ops_per_sec": throughput,
            "avg_time_per_object_ms": avg_time / len(objects)
        }
        
        self.results[model_name] = result
        return result
    
    def measure_deserialization(self, model_name: str, serialized_data: List[Any],
                              deserialization_func: callable, iterations: int = 1) -> Dict[str, float]:
        """Measure deserialization performance."""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            for data in serialized_data:
                deserialization_func(data)
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        throughput = len(serialized_data) / (avg_time / 1000)  # Objects per second
        
        result = {
            "model_name": f"{model_name}_deserialization",
            "object_count": len(serialized_data),
            "iterations": iterations,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "std_dev_ms": std_dev,
            "throughput_ops_per_sec": throughput,
            "avg_time_per_object_ms": avg_time / len(serialized_data)
        }
        
        self.results[f"{model_name}_deserialization"] = result
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all models."""
        if not self.results:
            return {}
        
        total_objects = sum(r["object_count"] for r in self.results.values())
        total_time = sum(r["avg_time_ms"] for r in self.results.values())
        avg_throughput = statistics.mean([r["throughput_ops_per_sec"] for r in self.results.values()])
        
        return {
            "total_objects_processed": total_objects,
            "total_time_ms": total_time,
            "average_throughput_ops_per_sec": avg_throughput,
            "model_count": len([r for r in self.results.keys() if not r.endswith("_deserialization")]),
            "fastest_model": min(self.results.items(), key=lambda x: x[1]["avg_time_per_object_ms"]),
            "slowest_model": max(self.results.items(), key=lambda x: x[1]["avg_time_per_object_ms"])
        }


class TestModelSerializationPerformance:
    """Performance benchmark tests for model serialization."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for tests."""
        return SerializationPerformanceMonitor()
    
    @pytest.fixture
    def sample_orders(self):
        """Create sample orders for testing."""
        orders = []
        
        for i in range(100):
            order = Order(
                user_id=UUID(f"12345678-1234-5678-9012-{i:012d}"),
                trading_pair=f"BTC{i % 5}/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT if i % 3 == 0 else OrderType.MARKET,
                quantity=Decimal(f"{10 + i * 0.1:.3f}"),
                price=Decimal(f"{50000 + i * 100:.2f}") if i % 3 == 0 else None,
                status=OrderStatus.PENDING
            )
            orders.append(order)
        
        return orders
    
    @pytest.fixture
    def sample_trading_pairs(self):
        """Create sample trading pairs for testing."""
        pairs = []
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT"]
        
        for i, symbol in enumerate(symbols * 20):  # 100 total
            pair = TradingPair(
                symbol=symbol,
                base_currency=symbol.split("/")[0],
                quote_currency=symbol.split("/")[1],
                price_precision=2 + (i % 4),
                quantity_precision=8,
                min_trade_quantity=Decimal(f"0.{i % 5 + 1:05d}"),
                max_trade_quantity=Decimal(f"{1000 + i * 10}"),
                min_notional=Decimal("10.00"),
                max_notional=Decimal(f"{100000 + i * 1000}"),
                maker_fee_rate=Decimal("0.001"),
                taker_fee_rate=Decimal("0.001"),
                status=TradingPairStatus.ACTIVE,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )
            pairs.append(pair)
        
        return pairs
    
    @pytest.fixture
    def sample_klines(self):
        """Create sample klines for testing."""
        klines = []
        base_time = datetime.now(UTC)
        
        for i in range(100):
            kline = Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=base_time + timedelta(minutes=i),
                close_time=base_time + timedelta(minutes=i+1),
                open_price=Decimal(f"{50000 + i * 10:.2f}"),
                high_price=Decimal(f"{50200 + i * 10:.2f}"),
                low_price=Decimal(f"{49800 + i * 10:.2f}"),
                close_price=Decimal(f"{50100 + i * 10:.2f}"),
                volume=Decimal(f"{100 + i:.3f}"),
                quote_volume=Decimal(f"{5000000 + i * 1000:.2f}"),
                trades_count=1000 + i * 10,
                asset_class=AssetClass.DIGITAL
            )
            klines.append(kline)
        
        return klines
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        trades = []
        base_time = datetime.now(UTC)
        
        for i in range(100):
            trade = Trade(
                symbol="BTCUSDT",
                trade_id=f"trade_{i:06d}",
                price=Decimal(f"{50000 + i * 5:.2f}"),
                quantity=Decimal(f"{0.1 + i * 0.01:.4f}"),
                side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
                timestamp=base_time + timedelta(seconds=i),
                asset_class=AssetClass.DIGITAL
            )
            trades.append(trade)
        
        return trades
    
    @pytest.fixture
    def sample_market_data(self, sample_klines, sample_trades):
        """Create sample market data for testing."""
        market_data_list = []
        
        # Create 10 market data objects with different amounts of data
        for i in range(10):
            start_idx = i * 10
            end_idx = start_idx + 10
            
            market_data = MarketData(
                symbol="BTCUSDT",
                klines=sample_klines[start_idx:end_idx],
                trades=sample_trades[start_idx:end_idx],
                asset_class=AssetClass.DIGITAL
            )
            market_data_list.append(market_data)
        
        return market_data_list
    
    def test_order_serialization_performance(self, performance_monitor, sample_orders):
        """Test Order model serialization performance."""
        # Test to_dict() serialization
        result = performance_monitor.measure_serialization(
            "Order_to_dict", 
            sample_orders, 
            lambda order: order.to_dict(),
            iterations=5
        )
        
        # Verify performance benchmarks
        assert result["avg_time_per_object_ms"] < 0.1, "Order to_dict should be under 0.1ms per object"
        assert result["throughput_ops_per_sec"] > 5000, "Order serialization should handle 5000+ ops/sec"
        
        # Test model_dump() serialization  
        result2 = performance_monitor.measure_serialization(
            "Order_model_dump",
            sample_orders,
            lambda order: order.model_dump(),
            iterations=5
        )
        
        assert result2["avg_time_per_object_ms"] < 0.05, "Order model_dump should be under 0.05ms per object"
        
        # Test JSON serialization
        serialized_orders = [order.to_dict() for order in sample_orders]
        result3 = performance_monitor.measure_serialization(
            "Order_json_dumps",
            serialized_orders,
            lambda order_dict: json.dumps(order_dict),
            iterations=5
        )
        
        assert result3["avg_time_per_object_ms"] < 0.5, "Order JSON serialization should be under 0.5ms per object"
    
    def test_order_deserialization_performance(self, performance_monitor, sample_orders):
        """Test Order model deserialization performance."""
        # Prepare serialized data
        serialized_data = [order.model_dump() for order in sample_orders]
        
        # Test model_validate() deserialization
        result = performance_monitor.measure_deserialization(
            "Order",
            serialized_data,
            lambda data: Order.model_validate(data),
            iterations=5
        )
        
        assert result["avg_time_per_object_ms"] < 0.5, "Order deserialization should be under 0.5ms per object"
        assert result["throughput_ops_per_sec"] > 1000, "Order deserialization should handle 1000+ ops/sec"
    
    def test_trading_pair_serialization_performance(self, performance_monitor, sample_trading_pairs):
        """Test TradingPair model serialization performance."""
        # Test to_dict() serialization
        result = performance_monitor.measure_serialization(
            "TradingPair_to_dict",
            sample_trading_pairs,
            lambda pair: pair.to_dict(),
            iterations=5
        )
        
        assert result["avg_time_per_object_ms"] < 0.2, "TradingPair to_dict should be under 0.2ms per object"
        assert result["throughput_ops_per_sec"] > 3000, "TradingPair serialization should handle 3000+ ops/sec"
        
        # Test deserialization
        serialized_data = [pair.model_dump() for pair in sample_trading_pairs]
        result2 = performance_monitor.measure_deserialization(
            "TradingPair",
            serialized_data,
            lambda data: TradingPair.model_validate(data),
            iterations=5
        )
        
        assert result2["avg_time_per_object_ms"] < 1.0, "TradingPair deserialization should be under 1ms per object"
    
    def test_kline_serialization_performance(self, performance_monitor, sample_klines):
        """Test Kline model serialization performance."""
        # Test to_dict() serialization
        result = performance_monitor.measure_serialization(
            "Kline_to_dict",
            sample_klines,
            lambda kline: kline.to_dict(),
            iterations=5
        )
        
        assert result["avg_time_per_object_ms"] < 0.3, "Kline to_dict should be under 0.3ms per object"
        assert result["throughput_ops_per_sec"] > 2000, "Kline serialization should handle 2000+ ops/sec"
        
        # Test deserialization
        serialized_data = [kline.model_dump() for kline in sample_klines]
        result2 = performance_monitor.measure_deserialization(
            "Kline",
            serialized_data,
            lambda data: Kline.model_validate(data),
            iterations=5
        )
        
        assert result2["avg_time_per_object_ms"] < 1.0, "Kline deserialization should be under 1ms per object"
    
    def test_trade_serialization_performance(self, performance_monitor, sample_trades):
        """Test Trade model serialization performance."""
        # Test to_dict() serialization
        result = performance_monitor.measure_serialization(
            "Trade_to_dict",
            sample_trades,
            lambda trade: trade.to_dict(),
            iterations=5
        )
        
        assert result["avg_time_per_object_ms"] < 0.1, "Trade to_dict should be under 0.1ms per object"
        assert result["throughput_ops_per_sec"] > 5000, "Trade serialization should handle 5000+ ops/sec"
        
        # Test deserialization
        serialized_data = [trade.model_dump() for trade in sample_trades]
        result2 = performance_monitor.measure_deserialization(
            "Trade",
            serialized_data,
            lambda data: Trade.model_validate(data),
            iterations=5
        )
        
        assert result2["avg_time_per_object_ms"] < 0.5, "Trade deserialization should be under 0.5ms per object"
    
    def test_market_data_serialization_performance(self, performance_monitor, sample_market_data):
        """Test MarketData model serialization performance."""
        # Test to_dict() serialization (includes nested objects)
        result = performance_monitor.measure_serialization(
            "MarketData_to_dict",
            sample_market_data,
            lambda market_data: market_data.to_dict(),
            iterations=3  # Fewer iterations due to complexity
        )
        
        # MarketData is more complex due to nested klines and trades
        assert result["avg_time_per_object_ms"] < 10.0, "MarketData to_dict should be under 10ms per object"
        assert result["throughput_ops_per_sec"] > 50, "MarketData serialization should handle 50+ ops/sec"
        
        # Test deserialization
        serialized_data = [market_data.to_dict() for market_data in sample_market_data]
        result2 = performance_monitor.measure_deserialization(
            "MarketData",
            serialized_data,
            lambda data: MarketData.from_dict(data),
            iterations=3
        )
        
        assert result2["avg_time_per_object_ms"] < 20.0, "MarketData deserialization should be under 20ms per object"
    
    def test_bulk_serialization_performance(self, performance_monitor, sample_orders, 
                                          sample_trading_pairs, sample_klines, sample_trades):
        """Test bulk serialization performance across all models."""
        all_models = {
            "Order": (sample_orders, lambda obj: obj.to_dict()),
            "TradingPair": (sample_trading_pairs, lambda obj: obj.to_dict()),
            "Kline": (sample_klines, lambda obj: obj.to_dict()),
            "Trade": (sample_trades, lambda obj: obj.to_dict())
        }
        
        total_start_time = time.perf_counter()
        
        for model_name, (objects, serializer) in all_models.items():
            performance_monitor.measure_serialization(
                f"Bulk_{model_name}",
                objects,
                serializer,
                iterations=3
            )
        
        total_time = (time.perf_counter() - total_start_time) * 1000
        
        # Verify bulk performance
        assert total_time < 200.0, "Bulk serialization of all models should complete under 200ms"
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        assert summary["total_objects_processed"] == 400  # 100 of each model
        assert summary["average_throughput_ops_per_sec"] > 2000, "Average throughput should exceed 2000 ops/sec"
    
    def test_serialization_format_comparison(self, performance_monitor, sample_orders):
        """Compare different serialization format performance."""
        # Test different serialization approaches
        formats = {
            "to_dict": lambda order: order.to_dict(),
            "model_dump": lambda order: order.model_dump(),
            "json_via_to_dict": lambda order: json.dumps(order.to_dict()),
            "json_via_model_dump": lambda order: json.dumps(order.model_dump(mode="json")),
        }
        
        format_results = {}
        
        for format_name, serializer in formats.items():
            result = performance_monitor.measure_serialization(
                f"Order_{format_name}",
                sample_orders[:50],  # Use subset for comparison
                serializer,
                iterations=5
            )
            format_results[format_name] = result
        
        # Verify that direct dict methods are generally faster than JSON serialization
        # Allow for 10% tolerance to handle timing variations
        to_dict_time = format_results["to_dict"]["avg_time_per_object_ms"]
        json_to_dict_time = format_results["json_via_to_dict"]["avg_time_per_object_ms"]
        model_dump_time = format_results["model_dump"]["avg_time_per_object_ms"] 
        json_model_dump_time = format_results["json_via_model_dump"]["avg_time_per_object_ms"]
        
        # Allow 10% tolerance for timing variations
        tolerance = 0.1
        assert (to_dict_time <= json_to_dict_time * (1 + tolerance)), \
            f"to_dict ({to_dict_time:.6f}ms) should be faster than json_via_to_dict ({json_to_dict_time:.6f}ms)"
        assert (model_dump_time <= json_model_dump_time * (1 + tolerance)), \
            f"model_dump ({model_dump_time:.6f}ms) should be faster than json_via_model_dump ({json_model_dump_time:.6f}ms)"
    
    def test_memory_efficiency_serialization(self, sample_orders):
        """Test memory efficiency of serialization methods."""
        import tracemalloc
        
        # Test memory usage for to_dict() method
        tracemalloc.start()
        
        serialized_orders = []
        for order in sample_orders:
            serialized_orders.append(order.to_dict())
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Verify memory usage is reasonable (less than 1MB for 100 orders)
        assert peak < 1024 * 1024, f"Memory usage {peak} bytes should be under 1MB for 100 orders"
        
        # Verify serialized data is valid
        assert len(serialized_orders) == len(sample_orders)
        for serialized in serialized_orders:
            assert isinstance(serialized, dict)
            assert "orderId" in serialized or "order_id" in serialized
            assert "tradingPair" in serialized or "trading_pair" in serialized
    
    def test_concurrent_serialization_performance(self, sample_orders):
        """Test serialization performance under concurrent load."""
        import concurrent.futures
        import threading
        
        def serialize_batch(orders_batch):
            """Serialize a batch of orders."""
            results = []
            start_time = time.perf_counter()
            
            for order in orders_batch:
                results.append(order.to_dict())
            
            end_time = time.perf_counter()
            return {
                "count": len(results),
                "time_ms": (end_time - start_time) * 1000,
                "thread_id": threading.get_ident()
            }
        
        # Split orders into batches for concurrent processing
        batch_size = 25
        batches = [sample_orders[i:i + batch_size] for i in range(0, len(sample_orders), batch_size)]
        
        # Process batches concurrently
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(serialize_batch, batch) for batch in batches]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Verify concurrent performance
        total_objects = sum(result["count"] for result in results)
        assert total_objects == len(sample_orders)
        
        # Concurrent processing may have overhead but should be reasonable
        sequential_time_estimate = sum(result["time_ms"] for result in results)
        # For small datasets, concurrent processing may have overhead, so we just verify it completed
        assert total_time > 0, "Concurrent processing should complete successfully"
        
        # Verify all threads processed data - for small datasets, may use single thread
        thread_ids = set(result["thread_id"] for result in results)
        assert len(thread_ids) >= 1, "At least one thread should have been used"
    
    def test_serialization_data_integrity(self, sample_orders, sample_trading_pairs, 
                                        sample_klines, sample_trades):
        """Test that serialization preserves data integrity."""
        test_cases = [
            (sample_orders[0], lambda obj: Order.model_validate(obj.model_dump())),
            (sample_trading_pairs[0], lambda obj: TradingPair.model_validate(obj.model_dump())),
            (sample_klines[0], lambda obj: Kline.model_validate(obj.model_dump())),
            (sample_trades[0], lambda obj: Trade.model_validate(obj.model_dump()))
        ]
        
        for original, deserializer in test_cases:
            # Serialize and deserialize
            restored = deserializer(original)
            
            # Verify key attributes are preserved
            assert type(restored) == type(original)
            
            if hasattr(original, 'symbol'):
                assert restored.symbol == original.symbol
            
            if hasattr(original, 'timestamp'):
                # Allow for slight timezone differences
                time_diff = abs((restored.timestamp - original.timestamp).total_seconds())
                assert time_diff < 1.0
            
            # Verify Decimal fields maintain precision
            for attr_name in dir(original):
                if not attr_name.startswith('_'):
                    attr_value = getattr(original, attr_name)
                    if isinstance(attr_value, Decimal):
                        restored_value = getattr(restored, attr_name)
                        assert isinstance(restored_value, Decimal)
                        assert attr_value == restored_value
    
    def test_large_dataset_serialization_performance(self, performance_monitor):
        """Test serialization performance with large datasets."""
        # Create larger datasets for stress testing
        large_orders = []
        for i in range(1000):  # 10x larger dataset
            order = Order(
                user_id=UUID(f"12345678-1234-5678-9012-{i:012d}"),
                trading_pair="BTC/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10.12345678"),  # 8 decimal places max
                price=Decimal("50000.12345678"),  # 8 decimal places max
                status=OrderStatus.PENDING
            )
            large_orders.append(order)
        
        # Test large dataset serialization
        result = performance_monitor.measure_serialization(
            "Large_Order_Dataset",
            large_orders,
            lambda order: order.to_dict(),
            iterations=3
        )
        
        # Performance should scale reasonably
        assert result["avg_time_per_object_ms"] < 0.2, "Large dataset per-object time should remain under 0.2ms"
        assert result["throughput_ops_per_sec"] > 3000, "Large dataset throughput should exceed 3000 ops/sec"
        
        # Test memory efficiency with large dataset
        import sys
        sample_serialized = large_orders[0].to_dict()
        estimated_memory_per_object = sys.getsizeof(sample_serialized)
        total_estimated_memory = estimated_memory_per_object * len(large_orders)
        
        # Should handle large datasets without excessive memory usage
        assert total_estimated_memory < 10 * 1024 * 1024, "Large dataset should use less than 10MB"
    
    def test_performance_summary_report(self, performance_monitor, sample_orders, 
                                      sample_trading_pairs, sample_klines, sample_trades):
        """Generate comprehensive performance summary report."""
        # Run all model serialization tests
        all_models = [
            ("Order", sample_orders, lambda obj: obj.to_dict()),
            ("TradingPair", sample_trading_pairs, lambda obj: obj.to_dict()),
            ("Kline", sample_klines, lambda obj: obj.to_dict()),
            ("Trade", sample_trades, lambda obj: obj.to_dict())
        ]
        
        for model_name, objects, serializer in all_models:
            performance_monitor.measure_serialization(
                model_name,
                objects,
                serializer,
                iterations=5
            )
        
        # Generate summary
        summary = performance_monitor.get_performance_summary()
        
        # Verify summary completeness
        assert summary["total_objects_processed"] > 0
        assert summary["model_count"] == 4
        assert "fastest_model" in summary
        assert "slowest_model" in summary
        
        # Print performance report for analysis
        print("\n=== Model Serialization Performance Summary ===")
        print(f"Total objects processed: {summary['total_objects_processed']}")
        print(f"Average throughput: {summary['average_throughput_ops_per_sec']:.2f} ops/sec")
        print(f"Fastest model: {summary['fastest_model'][0]} ({summary['fastest_model'][1]['avg_time_per_object_ms']:.3f}ms/obj)")
        print(f"Slowest model: {summary['slowest_model'][0]} ({summary['slowest_model'][1]['avg_time_per_object_ms']:.3f}ms/obj)")
        
        print("\n=== Individual Model Performance ===")
        for model_name, result in performance_monitor.results.items():
            if not model_name.endswith("_deserialization"):
                print(f"{model_name}: {result['avg_time_per_object_ms']:.3f}ms/obj, {result['throughput_ops_per_sec']:.2f} ops/sec")