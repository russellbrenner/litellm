import json

from litellm.llms.chatgpt.authenticator import Authenticator


CANONICAL = {
    "access_token": "access-top",
    "refresh_token": "refresh-top",
    "id_token": "id-top",
    "expires_at": 4102444800,
    "account_id": "acct-top",
}

LEGACY = {
    "tokens": {
        "access_token": "access-nested",
        "refresh_token": "refresh-nested",
        "id_token": "id-nested",
        "account_id": "acct-nested",
    },
    "expires_at": 4102444800,
}


def write_auth(tmp_path, payload, monkeypatch):
    auth_dir = tmp_path / "chatgpt"
    auth_dir.mkdir()
    auth_file = auth_dir / "auth.json"
    auth_file.write_text(json.dumps(payload))
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(auth_dir))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    return Authenticator()


def test_canonical_shape_reads_all_expected_fields(tmp_path, monkeypatch):
    auth = write_auth(tmp_path, CANONICAL, monkeypatch)
    auth_data = auth._read_auth_file()

    assert auth.get_access_token() == "access-top"
    assert auth.get_account_id() == "acct-top"
    assert auth_data is not None
    assert auth_data["access_token"] == "access-top"
    assert auth_data["refresh_token"] == "refresh-top"
    assert auth_data["id_token"] == "id-top"
    assert auth_data["expires_at"] == 4102444800
    assert auth_data["account_id"] == "acct-top"


def test_legacy_nested_shape_reads_all_expected_fields(tmp_path, monkeypatch):
    auth = write_auth(tmp_path, LEGACY, monkeypatch)
    auth_data = auth._read_auth_file()

    assert auth.get_access_token() == "access-nested"
    assert auth.get_account_id() == "acct-nested"
    assert auth_data is not None
    assert auth_data["access_token"] == "access-nested"
    assert auth_data["refresh_token"] == "refresh-nested"
    assert auth_data["id_token"] == "id-nested"
    assert auth_data["expires_at"] == 4102444800
    assert auth_data["account_id"] == "acct-nested"
