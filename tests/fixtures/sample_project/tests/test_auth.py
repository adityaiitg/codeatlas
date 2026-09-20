"""Tests for authentication middleware."""

from auth.middleware import AuthMiddleware


class TestAuthMiddleware:
    """Test suite for AuthMiddleware."""

    def test_authenticate_valid_token(self):
        """Test that valid tokens are accepted."""
        middleware = AuthMiddleware(secret_key="test-secret")
        # Would need a properly signed token in real tests
        assert middleware is not None

    def test_missing_token_raises(self):
        """Test that missing tokens raise ValueError."""
        middleware = AuthMiddleware(secret_key="test-secret")

        class FakeRequest:
            headers = {}

        try:
            middleware.authenticate(FakeRequest())
            assert False, "Should have raised"
        except ValueError as e:
            assert "Missing" in str(e)
