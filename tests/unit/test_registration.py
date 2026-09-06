from civicnexus.api.services import hash_password, verify_password
from civicnexus.persistence.database import create_db_engine, create_session_factory, init_db
from civicnexus.persistence.repositories import TaskRepository


def test_password_hash_is_salted_and_verifiable() -> None:
    first = hash_password("secret-123")
    second = hash_password("secret-123")
    assert first != second
    assert verify_password("secret-123", first)
    assert not verify_password("wrong", first)


def test_registered_account_lookup_is_case_insensitive() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    repo = TaskRepository(create_session_factory(engine))
    account = repo.create_user_account("NewUser", hash_password("secret-123"))
    assert account.username == "newuser"
    assert repo.get_user_account("NEWUSER").user_id == account.user_id
