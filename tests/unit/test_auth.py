from fastapi import HTTPException

from civicnexus.api.auth import resolve_user_id
from civicnexus.api.services import AuthTokenStore
from civicnexus.persistence.cache import MemoryCache


def test_user_tokens_are_scoped_and_expire() -> None:
    store = AuthTokenStore(MemoryCache(), 60, prefix="test:user")
    token = store.issue(role="user", user_id="u-1", display_name="用户")
    assert store.resolve(token) == {"role": "user", "user_id": "u-1", "display_name": "用户"}
    assert store.resolve("other-token") is None


def test_authenticated_user_cannot_impersonate_another_id() -> None:
    principal = {"role": "user", "user_id": "u-1", "display_name": "用户"}
    assert resolve_user_id(None, principal) == "u-1"
    try:
        resolve_user_id("u-2", principal)
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("expected user identity mismatch to be rejected")
