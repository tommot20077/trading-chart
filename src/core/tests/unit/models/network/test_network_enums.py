# ABOUTME: Unit tests for network enumeration classes including ConnectionStatus
# ABOUTME: Tests cover normal cases, exception cases, and boundary cases following TDD principles

import pytest

from core.models.network.enum import ConnectionStatus


class TestConnectionStatus:
    """Test cases for ConnectionStatus enum."""

    @pytest.mark.unit
    def test_connection_status_values_normal_case(self):
        """Test that ConnectionStatus enum has correct string values."""
        assert ConnectionStatus.CONNECTING == "connecting"
        assert ConnectionStatus.CONNECTED == "connected"
        assert ConnectionStatus.DISCONNECTED == "disconnected"
        assert ConnectionStatus.RECONNECTING == "reconnecting"
        assert ConnectionStatus.ERROR == "error"

    @pytest.mark.unit
    def test_connection_status_string_representation(self):
        """Test string representation of ConnectionStatus enum members."""
        assert str(ConnectionStatus.CONNECTING) == "ConnectionStatus.CONNECTING"
        assert str(ConnectionStatus.CONNECTED) == "ConnectionStatus.CONNECTED"
        assert str(ConnectionStatus.DISCONNECTED) == "ConnectionStatus.DISCONNECTED"
        assert str(ConnectionStatus.RECONNECTING) == "ConnectionStatus.RECONNECTING"
        assert str(ConnectionStatus.ERROR) == "ConnectionStatus.ERROR"

    @pytest.mark.unit
    def test_connection_status_equality(self):
        """Test equality comparison of ConnectionStatus enum members."""
        assert ConnectionStatus.CONNECTING == ConnectionStatus.CONNECTING
        assert ConnectionStatus.CONNECTED == ConnectionStatus.CONNECTED
        assert ConnectionStatus.CONNECTING != ConnectionStatus.CONNECTED
        assert ConnectionStatus.DISCONNECTED != ConnectionStatus.RECONNECTING
        assert ConnectionStatus.ERROR != ConnectionStatus.CONNECTED

    @pytest.mark.unit
    def test_connection_status_membership(self):
        """Test membership checking for ConnectionStatus enum."""
        status_values = [status.value for status in ConnectionStatus]
        assert "connecting" in status_values
        assert "connected" in status_values
        assert "disconnected" in status_values
        assert "reconnecting" in status_values
        assert "error" in status_values
        assert "invalid_status" not in status_values

    @pytest.mark.unit
    def test_connection_status_iteration(self):
        """Test iteration over ConnectionStatus enum members."""
        statuses = list(ConnectionStatus)
        assert len(statuses) == 5
        assert ConnectionStatus.CONNECTING in statuses
        assert ConnectionStatus.CONNECTED in statuses
        assert ConnectionStatus.DISCONNECTED in statuses
        assert ConnectionStatus.RECONNECTING in statuses
        assert ConnectionStatus.ERROR in statuses

    @pytest.mark.unit
    def test_connection_status_from_string(self):
        """Test creating ConnectionStatus from string value."""
        assert ConnectionStatus("connecting") == ConnectionStatus.CONNECTING
        assert ConnectionStatus("connected") == ConnectionStatus.CONNECTED
        assert ConnectionStatus("disconnected") == ConnectionStatus.DISCONNECTED
        assert ConnectionStatus("reconnecting") == ConnectionStatus.RECONNECTING
        assert ConnectionStatus("error") == ConnectionStatus.ERROR

    @pytest.mark.unit
    def test_connection_status_invalid_value_raises_exception(self):
        """Test that invalid connection status value raises ValueError."""
        with pytest.raises(ValueError):
            ConnectionStatus("invalid_status")

        with pytest.raises(ValueError):
            ConnectionStatus("CONNECTED")  # Case sensitive

        with pytest.raises(ValueError):
            ConnectionStatus("")

    @pytest.mark.unit
    def test_connection_status_completeness(self):
        """Test that all expected connection statuses are present."""
        expected_statuses = {"connecting", "connected", "disconnected", "reconnecting", "error"}
        actual_statuses = {status.value for status in ConnectionStatus}
        assert actual_statuses == expected_statuses

    @pytest.mark.unit
    def test_connection_status_lifecycle_logic(self):
        """Test logical relationships in connection lifecycle."""
        # Active states (connection is being attempted or established)
        active_states = [ConnectionStatus.CONNECTING, ConnectionStatus.CONNECTED, ConnectionStatus.RECONNECTING]
        for state in active_states:
            assert state.value in ["connecting", "connected", "reconnecting"]

        # Inactive states (connection is not active)
        inactive_states = [ConnectionStatus.DISCONNECTED, ConnectionStatus.ERROR]
        for state in inactive_states:
            assert state.value in ["disconnected", "error"]

        # Transitional states (temporary states during connection changes)
        transitional_states = [ConnectionStatus.CONNECTING, ConnectionStatus.RECONNECTING]
        for state in transitional_states:
            assert state.value in ["connecting", "reconnecting"]

    @pytest.mark.unit
    def test_connection_status_repr_representation(self):
        """Test repr representation of ConnectionStatus enum members."""
        assert "ConnectionStatus.CONNECTING" in repr(ConnectionStatus.CONNECTING)
        assert "ConnectionStatus.CONNECTED" in repr(ConnectionStatus.CONNECTED)
        assert "ConnectionStatus.DISCONNECTED" in repr(ConnectionStatus.DISCONNECTED)
        assert "ConnectionStatus.RECONNECTING" in repr(ConnectionStatus.RECONNECTING)
        assert "ConnectionStatus.ERROR" in repr(ConnectionStatus.ERROR)

    @pytest.mark.unit
    def test_connection_status_hash_consistency(self):
        """Test that ConnectionStatus enum members are hashable and consistent."""
        # Test that enum members can be used as dictionary keys
        status_handlers = {
            ConnectionStatus.CONNECTING: "handle_connecting",
            ConnectionStatus.CONNECTED: "handle_connected",
            ConnectionStatus.DISCONNECTED: "handle_disconnected",
            ConnectionStatus.RECONNECTING: "handle_reconnecting",
            ConnectionStatus.ERROR: "handle_error",
        }

        assert len(status_handlers) == 5
        assert status_handlers[ConnectionStatus.CONNECTING] == "handle_connecting"
        assert status_handlers[ConnectionStatus.CONNECTED] == "handle_connected"

        # Test that hash is consistent
        assert hash(ConnectionStatus.CONNECTING) == hash(ConnectionStatus.CONNECTING)
        assert hash(ConnectionStatus.CONNECTING) != hash(ConnectionStatus.CONNECTED)

    @pytest.mark.unit
    def test_connection_status_set_operations(self):
        """Test ConnectionStatus enum members in set operations."""
        status_set = {ConnectionStatus.CONNECTING, ConnectionStatus.CONNECTED, ConnectionStatus.CONNECTING}
        assert len(status_set) == 2  # Duplicates removed
        assert ConnectionStatus.CONNECTING in status_set
        assert ConnectionStatus.CONNECTED in status_set

        # Test set operations for business logic
        healthy_states = {ConnectionStatus.CONNECTING, ConnectionStatus.CONNECTED, ConnectionStatus.RECONNECTING}
        unhealthy_states = {ConnectionStatus.DISCONNECTED, ConnectionStatus.ERROR}

        # No overlap between healthy and unhealthy states
        assert healthy_states.intersection(unhealthy_states) == set()

        # Union should contain all states
        all_states = healthy_states.union(unhealthy_states)
        assert len(all_states) == 5

    @pytest.mark.unit
    def test_connection_status_state_transitions(self):
        """Test logical state transition possibilities."""
        # Define possible transitions (this tests business logic understanding)
        valid_transitions = {
            ConnectionStatus.DISCONNECTED: [ConnectionStatus.CONNECTING],
            ConnectionStatus.CONNECTING: [
                ConnectionStatus.CONNECTED,
                ConnectionStatus.ERROR,
                ConnectionStatus.DISCONNECTED,
            ],
            ConnectionStatus.CONNECTED: [
                ConnectionStatus.DISCONNECTED,
                ConnectionStatus.ERROR,
                ConnectionStatus.RECONNECTING,
            ],
            ConnectionStatus.RECONNECTING: [
                ConnectionStatus.CONNECTED,
                ConnectionStatus.ERROR,
                ConnectionStatus.DISCONNECTED,
            ],
            ConnectionStatus.ERROR: [ConnectionStatus.CONNECTING, ConnectionStatus.DISCONNECTED],
        }

        # Verify that all states are accounted for
        assert set(valid_transitions.keys()) == set(ConnectionStatus)

        # Verify that all transition targets are valid ConnectionStatus values
        for source, targets in valid_transitions.items():
            assert isinstance(source, ConnectionStatus)
            for target in targets:
                assert isinstance(target, ConnectionStatus)

    @pytest.mark.unit
    def test_connection_status_boundary_cases(self):
        """Test boundary cases for ConnectionStatus enum."""
        # Test that we can iterate multiple times
        first_iteration = list(ConnectionStatus)
        second_iteration = list(ConnectionStatus)
        assert first_iteration == second_iteration

        # Test that enum members are singleton
        assert ConnectionStatus.CONNECTED is ConnectionStatus.CONNECTED
        assert ConnectionStatus("connected") is ConnectionStatus.CONNECTED
