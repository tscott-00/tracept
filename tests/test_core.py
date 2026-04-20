import pytest

def test_exceptions():
    # Biscuit detection (baking twice should raise an error)
    with pytest.raises(, match='.*Biscuit.*':

