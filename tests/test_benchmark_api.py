import pytest


@pytest.mark.skip(reason="모델 로딩 비용이 커서 기본 테스트에서 제외")
def test_placeholder() -> None:
    assert True
