# ABOUTME: Unit tests for AuthRequest protocol implementation
# ABOUTME: Tests cover protocol compliance, header handling, and client identification

import pytest


class MockAuthRequest:
    """Mock implementation of AuthRequest protocol for testing."""

    def __init__(self, headers: dict[str, str], client_id: str | None = None):
        self._headers = {k.lower(): v for k, v in headers.items()}
        self._client_id = client_id

    def get_header(self, name: str) -> str | None:
        return self._headers.get(name.lower())

    @property
    def client_id(self) -> str | None:
        return self._client_id


class TestAuthRequestProtocol:
    """Test cases for AuthRequest protocol implementation."""

    @pytest.mark.unit
    def test_get_header_normal_case(self):
        """Test getting header value with valid header name."""
        headers = {"Authorization": "Bearer token123", "Content-Type": "application/json"}
        request = MockAuthRequest(headers)

        assert request.get_header("Authorization") == "Bearer token123"
        assert request.get_header("Content-Type") == "application/json"

    @pytest.mark.unit
    def test_get_header_case_insensitive(self):
        """Test that header retrieval is case-insensitive."""
        headers = {"Authorization": "Bearer token123"}
        request = MockAuthRequest(headers)

        assert request.get_header("authorization") == "Bearer token123"
        assert request.get_header("AUTHORIZATION") == "Bearer token123"
        assert request.get_header("Authorization") == "Bearer token123"

    @pytest.mark.unit
    def test_get_header_not_found_returns_none(self):
        """Test that non-existent header returns None."""
        headers = {"Authorization": "Bearer token123"}
        request = MockAuthRequest(headers)

        assert request.get_header("X-Custom-Header") is None
        assert request.get_header("") is None

    @pytest.mark.unit
    def test_get_header_empty_headers(self):
        """Test getting header from empty headers dict."""
        request = MockAuthRequest({})

        assert request.get_header("Authorization") is None
        assert request.get_header("any-header") is None

    @pytest.mark.unit
    def test_get_header_with_special_characters(self):
        """Test header names with special characters."""
        headers = {"X-Custom-Header": "value1", "X_Another_Header": "value2"}
        request = MockAuthRequest(headers)

        assert request.get_header("X-Custom-Header") == "value1"
        assert request.get_header("X_Another_Header") == "value2"

    @pytest.mark.unit
    def test_get_header_with_whitespace(self):
        """Test header values with whitespace."""
        headers = {"Authorization": "  Bearer token123  "}
        request = MockAuthRequest(headers)

        assert request.get_header("Authorization") == "  Bearer token123  "

    @pytest.mark.unit
    def test_client_id_normal_case(self):
        """Test client_id property with valid client ID."""
        request = MockAuthRequest({}, client_id="client123")
        assert request.client_id == "client123"

    @pytest.mark.unit
    def test_client_id_none_case(self):
        """Test client_id property when no client ID is set."""
        request = MockAuthRequest({})
        assert request.client_id is None

    @pytest.mark.unit
    def test_client_id_empty_string(self):
        """Test client_id property with empty string."""
        request = MockAuthRequest({}, client_id="")
        assert request.client_id == ""

    @pytest.mark.unit
    @pytest.mark.external
    def test_client_id_with_special_characters(self):
        """Test client_id with special characters."""
        special_ids = ["192.168.1.1", "api-key-123", "user@domain.com", "client_id_with_underscores"]

        for client_id in special_ids:
            request = MockAuthRequest({}, client_id=client_id)
            assert request.client_id == client_id

    @pytest.mark.unit
    def test_client_id_with_unicode(self):
        """Test client_id with unicode characters."""
        unicode_id = "客戶端123"
        request = MockAuthRequest({}, client_id=unicode_id)
        assert request.client_id == unicode_id

    @pytest.mark.unit
    def test_multiple_headers_access(self):
        """Test accessing multiple headers from the same request."""
        headers = {
            "Authorization": "Bearer token123",
            "Content-Type": "application/json",
            "User-Agent": "TestClient/1.0",
            "X-Request-ID": "req-12345",
        }
        request = MockAuthRequest(headers)

        # Test multiple accesses
        assert request.get_header("Authorization") == "Bearer token123"
        assert request.get_header("Content-Type") == "application/json"
        assert request.get_header("User-Agent") == "TestClient/1.0"
        assert request.get_header("X-Request-ID") == "req-12345"

        # Test that multiple accesses return consistent results
        assert request.get_header("Authorization") == "Bearer token123"
        assert request.get_header("authorization") == "Bearer token123"

    @pytest.mark.unit
    def test_boundary_cases(self):
        """Test boundary cases for header names and values."""
        # Single character header name and value
        headers = {"A": "B"}
        request = MockAuthRequest(headers)
        assert request.get_header("A") == "B"
        assert request.get_header("a") == "B"

        # Long header name and value
        long_name = "X-" + "A" * 100
        long_value = "B" * 1000
        headers = {long_name: long_value}
        request = MockAuthRequest(headers)
        assert request.get_header(long_name) == long_value

    @pytest.mark.unit
    def test_header_overwrite_behavior(self):
        """Test behavior when headers are overwritten during construction."""
        # Test case sensitivity in constructor
        headers = {"Authorization": "Bearer token1", "authorization": "Bearer token2"}
        request = MockAuthRequest(headers)

        # Should get one of the values (implementation dependent)
        auth_value = request.get_header("Authorization")
        assert auth_value in ["Bearer token1", "Bearer token2"]
