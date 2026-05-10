import juliams


def test_public_all_exports_exist():
    missing = [name for name in juliams.__all__ if not hasattr(juliams, name)]

    assert missing == []
