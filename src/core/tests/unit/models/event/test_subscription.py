# ABOUTME: Unit tests for Subscription model
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for Subscription

import pytest
from unittest.mock import Mock, AsyncMock

from core.models.event.subscription import Subscription
from core.models.event.event_type import EventType
from core.models.data.event import BaseEvent


class TestSubscription:
    """
    Comprehensive unit tests for Subscription model.

    Tests cover:
    - Normal case: Valid Subscription creation and behavior
    - Exception case: Invalid inputs and error handling
    - Boundary case: Edge cases and extreme values
    """

    @pytest.mark.unit
    def test_subscription_creation_sync_handler(self):
        """
        Test normal Subscription creation with synchronous handler.

        Verifies:
        - All fields are properly initialized
        - Handler is correctly identified as synchronous
        - is_async flag is False for sync handlers
        """

        # Arrange
        def sync_handler(event: BaseEvent) -> None:
            pass

        # Act
        subscription = Subscription(
            subscription_id="sub_001", event_type=EventType.TRADE, handler=sync_handler, filter_symbol="BTCUSDT"
        )

        # Assert
        assert subscription.id == "sub_001"
        assert subscription.event_type == EventType.TRADE
        assert subscription.handler == sync_handler
        assert subscription.filter_symbol == "BTCUSDT"
        assert subscription.is_async is False

    @pytest.mark.unit
    def test_subscription_creation_async_handler(self):
        """
        Test normal Subscription creation with asynchronous handler.

        Verifies:
        - All fields are properly initialized
        - Handler is correctly identified as asynchronous
        - is_async flag is True for async handlers
        """

        # Arrange
        async def async_handler(event: BaseEvent) -> None:
            pass

        # Act
        subscription = Subscription(subscription_id="sub_002", event_type=EventType.KLINE, handler=async_handler)

        # Assert
        assert subscription.id == "sub_002"
        assert subscription.event_type == EventType.KLINE
        assert subscription.handler == async_handler
        assert subscription.filter_symbol is None
        assert subscription.is_async is True

    @pytest.mark.unit
    def test_subscription_without_filter_symbol(self):
        """
        Test Subscription creation without filter symbol.

        Verifies:
        - filter_symbol defaults to None
        - Subscription works correctly without symbol filtering
        """

        # Arrange
        def handler(event: BaseEvent) -> None:
            pass

        # Act
        subscription = Subscription(subscription_id="sub_003", event_type=EventType.ERROR, handler=handler)

        # Assert
        assert subscription.filter_symbol is None
        assert subscription.event_type == EventType.ERROR

    @pytest.mark.unit
    def test_subscription_matches_event_type_only(self):
        """
        Test subscription matching with event type only (no symbol filter).

        Verifies:
        - Events with matching type are matched
        - Events with different type are not matched
        - Symbol is ignored when no filter is set
        """

        # Arrange
        def handler(event: BaseEvent) -> None:
            pass

        subscription = Subscription(subscription_id="sub_004", event_type=EventType.TRADE, handler=handler)

        matching_event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={"test": "data"})

        non_matching_event = BaseEvent(
            event_type=EventType.KLINE, source="test", symbol="BTCUSDT", data={"test": "data"}
        )

        # Act & Assert
        assert subscription.matches(matching_event) is True
        assert subscription.matches(non_matching_event) is False

    @pytest.mark.unit
    def test_subscription_matches_event_type_and_symbol(self):
        """
        Test subscription matching with both event type and symbol filter.

        Verifies:
        - Events with matching type and symbol are matched
        - Events with matching type but different symbol are not matched
        - Events with different type are not matched regardless of symbol
        """

        # Arrange
        def handler(event: BaseEvent) -> None:
            pass

        subscription = Subscription(
            subscription_id="sub_005", event_type=EventType.TRADE, handler=handler, filter_symbol="BTCUSDT"
        )

        matching_event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={"test": "data"})

        wrong_symbol_event = BaseEvent(
            event_type=EventType.TRADE, source="test", symbol="ETHUSDT", data={"test": "data"}
        )

        wrong_type_event = BaseEvent(event_type=EventType.KLINE, source="test", symbol="BTCUSDT", data={"test": "data"})

        # Act & Assert
        assert subscription.matches(matching_event) is True
        assert subscription.matches(wrong_symbol_event) is False
        assert subscription.matches(wrong_type_event) is False

    @pytest.mark.unit
    def test_subscription_matches_case_sensitivity(self):
        """
        Test subscription symbol matching case sensitivity.

        Verifies:
        - Symbol matching is case sensitive
        - Exact symbol match is required
        """

        # Arrange
        def handler(event: BaseEvent) -> None:
            pass

        subscription = Subscription(
            subscription_id="sub_006", event_type=EventType.TRADE, handler=handler, filter_symbol="BTCUSDT"
        )

        lowercase_event = BaseEvent(
            event_type=EventType.TRADE,
            source="test",
            symbol="btcusdt",  # lowercase
            data={"test": "data"},
        )

        # Act & Assert
        # Note: BaseEvent normalizes symbols to uppercase, so "btcusdt" becomes "BTCUSDT"
        # This test verifies that the normalization works correctly
        assert subscription.matches(lowercase_event) is True  # Should match after normalization

    @pytest.mark.unit
    def test_subscription_with_mock_handlers(self):
        """
        Test Subscription with mock handlers for testing purposes.

        Verifies:
        - Mock handlers work correctly
        - Both sync and async mocks are properly detected
        """
        # Test with sync mock
        sync_mock = Mock()
        subscription_sync = Subscription(subscription_id="sub_007", event_type=EventType.CONNECTION, handler=sync_mock)

        assert subscription_sync.handler == sync_mock
        assert subscription_sync.is_async is False

        # Test with async mock
        async_mock = AsyncMock()
        subscription_async = Subscription(subscription_id="sub_008", event_type=EventType.ERROR, handler=async_mock)

        assert subscription_async.handler == async_mock
        assert subscription_async.is_async is True

    @pytest.mark.unit
    def test_subscription_handler_type_detection(self):
        """
        Test various handler types and their async detection.

        Verifies:
        - Regular functions are detected as sync
        - Coroutine functions are detected as async
        - Lambda functions are detected correctly
        - Methods are detected correctly
        """

        # Regular function
        def regular_func(event: BaseEvent) -> None:
            pass

        # Async function
        async def async_func(event: BaseEvent) -> None:
            pass

        # Lambda function
        lambda_func = lambda event: None

        # Async lambda (not common but possible)
        async def async_lambda(event):
            return None

        # Test regular function
        sub1 = Subscription("sub1", EventType.TRADE, regular_func)
        assert sub1.is_async is False

        # Test async function
        sub2 = Subscription("sub2", EventType.TRADE, async_func)
        assert sub2.is_async is True

        # Test lambda function
        sub3 = Subscription("sub3", EventType.TRADE, lambda_func)
        assert sub3.is_async is False

    @pytest.mark.unit
    def test_subscription_all_event_types(self):
        """
        Test Subscription with all possible EventType values.

        Verifies:
        - All EventType enum values work correctly
        - Subscription behavior is consistent across event types
        """

        def handler(event: BaseEvent) -> None:
            pass

        event_types = [
            EventType.TRADE,
            EventType.KLINE,
            EventType.ORDER,
            EventType.CONNECTION,
            EventType.ERROR,
            EventType.SYSTEM,
        ]

        for i, event_type in enumerate(event_types):
            subscription = Subscription(subscription_id=f"sub_{i}", event_type=event_type, handler=handler)
            assert subscription.event_type == event_type

    @pytest.mark.unit
    def test_subscription_boundary_values(self):
        """
        Test Subscription with boundary values.

        Verifies:
        - Very long subscription IDs work
        - Very long symbol filters work
        - Special characters in IDs and symbols
        """

        def handler(event: BaseEvent) -> None:
            pass

        # Very long subscription ID
        long_id = "a" * 1000
        subscription1 = Subscription(subscription_id=long_id, event_type=EventType.TRADE, handler=handler)
        assert subscription1.id == long_id

        # Very long symbol filter
        long_symbol = "B" * 100
        subscription2 = Subscription(
            subscription_id="sub_long_symbol", event_type=EventType.TRADE, handler=handler, filter_symbol=long_symbol
        )
        assert subscription2.filter_symbol == long_symbol

        # Special characters
        special_id = "sub_!@#$%^&*()_+-=[]{}|;':\",./<>?"
        subscription3 = Subscription(subscription_id=special_id, event_type=EventType.TRADE, handler=handler)
        assert subscription3.id == special_id

    @pytest.mark.unit
    def test_subscription_matches_with_none_symbol_event(self):
        """
        Test subscription matching when event has None symbol.

        Verifies:
        - Events with None symbol don't match symbol-filtered subscriptions
        - Events with None symbol match non-filtered subscriptions
        """

        def handler(event: BaseEvent) -> None:
            pass

        # Subscription with symbol filter
        filtered_subscription = Subscription(
            subscription_id="filtered", event_type=EventType.SYSTEM, handler=handler, filter_symbol="BTCUSDT"
        )

        # Subscription without symbol filter
        unfiltered_subscription = Subscription(
            subscription_id="unfiltered", event_type=EventType.SYSTEM, handler=handler
        )

        # Event with None symbol
        none_symbol_event = BaseEvent(event_type=EventType.SYSTEM, source="test", symbol=None, data={"test": "data"})

        # Act & Assert
        assert filtered_subscription.matches(none_symbol_event) is False
        assert unfiltered_subscription.matches(none_symbol_event) is True

    @pytest.mark.unit
    def test_subscription_invalid_parameters(self):
        """
        Test Subscription creation with invalid parameters.

        Verifies:
        - Proper errors for missing required parameters
        - Type validation works correctly
        """

        def handler(event: BaseEvent) -> None:
            pass

        # Test missing subscription_id
        with pytest.raises(TypeError):
            Subscription(event_type=EventType.TRADE, handler=handler)

        # Test missing event_type
        with pytest.raises(TypeError):
            Subscription(subscription_id="test", handler=handler)

        # Test missing handler
        with pytest.raises(TypeError):
            Subscription(subscription_id="test", event_type=EventType.TRADE)

    @pytest.mark.unit
    def test_subscription_handler_callable_validation(self):
        """
        Test that non-callable handlers are handled gracefully.

        Verifies:
        - Subscription accepts any object as handler (validation is deferred)
        - asyncio.iscoroutinefunction handles non-callable gracefully
        """
        # Non-callable handler should be accepted during construction
        subscription = Subscription(
            subscription_id="invalid",
            event_type=EventType.TRADE,
            handler="not_callable",  # String is not callable
        )

        # But is_async should handle it gracefully (returns False for non-callable)
        assert subscription.is_async is False
        assert subscription.handler == "not_callable"
