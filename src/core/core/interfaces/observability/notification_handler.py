# ABOUTME: Abstract notification handler interface for alert and notification management
# ABOUTME: Defines the contract for components that send notifications through various channels

from abc import abstractmethod, ABC

from core.models.types import AlertData


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
    async def send_notification(self, alert_data: AlertData) -> tuple[bool, str]:
        """
        Send a notification for the provided alert data.

        This method should implement the specific notification logic
        and return whether the notification was sent successfully.
        The method is designed to be non-blocking and return immediately.
        For retry logic, implementations should handle this internally
        or delegate to higher-level retry mechanisms.

        Args:
            alert_data (dict[str, Any]): Dictionary containing alert information including:
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
            tuple[bool, str]: A tuple where the first element indicates whether
                the notification was sent successfully, and the second element
                provides additional information about the result.

        Raises:
            NotificationError: If the notification fails due to configuration or
                network issues.
        """
        pass
