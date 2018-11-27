import pytest
from birdcloud.birdcloud import BirdCloud

def test_open_nonexistent_file():
    with pytest.raises(Exception):
        b = BirdCloud()
        b.from_raw_knmi_file('')

def test_open_existing_file():
    with pytest.raises(Exception):
        b = BirdCloud()
        b.from_raw_knmi_file('../data/raw/RAD_NL62_VOL_NA_201802010000.h5')
