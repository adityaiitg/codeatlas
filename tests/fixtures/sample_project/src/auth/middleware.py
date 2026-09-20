"""Authentication middleware."""

from auth.token import parse_jwt, verify_signature


class AuthMiddleware:
    """Validates incoming requests using JWT tokens."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def authenticate(self, request) -> dict:
        """Authenticate an incoming request.

        Extracts the JWT token from the Authorization header,
        validates it, and returns the decoded payload.
        """
        token = self._extract_token(request)
        if not token:
            raise ValueError("Missing authorization token")

        payload = parse_jwt(token)
        if not verify_signature(token, self.secret_key):
            raise ValueError("Invalid token signature")

        return payload

    def _extract_token(self, request) -> str | None:
        """Extract Bearer token from request headers."""
        auth_header = getattr(request, "headers", {}).get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None
