"""JWT token utilities."""

import hashlib
import json
import base64


def parse_jwt(token: str) -> dict:
    """Parse a JWT token and return the decoded payload."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")

    payload_b64 = parts[1]
    # Add padding
    padding = 4 - len(payload_b64) % 4
    payload_b64 += "=" * padding

    payload_json = base64.urlsafe_b64decode(payload_b64)
    return json.loads(payload_json)


def verify_signature(token: str, secret_key: str) -> bool:
    """Verify the JWT token signature using HMAC-SHA256."""
    parts = token.split(".")
    if len(parts) != 3:
        return False

    message = f"{parts[0]}.{parts[1]}"
    expected = hashlib.sha256(f"{message}{secret_key}".encode()).hexdigest()
    return parts[2] == expected
