from abc import abstractmethod, ABC
from typing import Any


class AbstractNotificationHandler(ABC):
    """
    [L0] Abstract base class for implementing notification handlers.

    This interface defines the contract for implementing notification functionality.
    Concrete implementations should provide specific notification channels
    (email, slack, webhook, etc.).

    Architecture note: This is a [L0] interface that has no dependencies on other
    asset_core modules and provides clean abstractions for notification implementations.
    """

    @abstractmethod
    async def send_notification(self, alert_data: dict[str, Any]) -> tuple[bool, str]:
        """
        Send a notification for the provided alert data.

        This method should implement the specific notification logic
        and return whether the notification was sent successfully.

        Args:
            alert_data: Dictionary containing alert information including:
                - id: Alert identifier
                - rule_name: Name of the triggered rule
                - severity: Alert severity level
                - status: Alert status
                - message: Alert message
                - labels: Alert labels
                - annotations: Alert annotations
                - fired_at: When the alert was fired
                - trace_id: Trace identifier

        Returns:
            Tuple of (success: bool, message: str) where success indicates
            whether the notification was sent successfully and message provides
            additional information about the result.

        Note:
            This method should be non-blocking and return immediately.
            For retry logic, implementations should handle this internally
            or delegate to higher-level retry mechanisms.
        """
        pass
