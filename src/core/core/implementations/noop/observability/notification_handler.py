# ABOUTME: No-operation implementation of AbstractNotificationHandler for testing
# ABOUTME: Provides minimal notification handling that always succeeds without side effects


from core.interfaces.observability.notification_handler import AbstractNotificationHandler
from core.models.types import AlertData


class NoOpNotificationHandler(AbstractNotificationHandler):
    """
    No-operation implementation of AbstractNotificationHandler.

    This implementation provides a minimal notification handler that always
    returns success without performing any actual notification operations.
    It's useful for testing, performance benchmarking, or when notification
    functionality is not required.

    Features:
    - Always returns successful notification sending
    - No actual notification processing or storage
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where notifications should be silenced
    - Performance benchmarking without notification overhead
    - Development environments where notifications are not needed
    - Fallback when notification systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation notification handler."""
        # No initialization needed for NoOp implementation
        pass

    async def send_notification(self, alert_data: AlertData) -> tuple[bool, str]:
        """
        Send a notification for the provided alert data.

        This implementation always returns success without performing any
        actual notification operations.

        Args:
            alert_data: Dictionary containing alert information (ignored)

        Returns:
            Tuple of (True, "NoOp notification sent successfully")
        """
        # Always return success for NoOp implementation
        return True, "NoOp notification sent successfully"
