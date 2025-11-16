import unittest
from pathlib import Path
import shutil

from .data import _find_band_path

class TestFindBandPath(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("test_tile_data")
        self.test_dir.mkdir(exist_ok=True)

        # Create a nested directory structure
        self.nested_dir = self.test_dir / "nested" / "sub_nested"
        self.nested_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy band file inside the nested directory
        self.band_file = self.nested_dir / "S2A_MSIL1C_B02.jp2"
        self.band_file.touch()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_find_band_path_recursive(self):
        # Test if _find_band_path can find the band file recursively
        found_path = _find_band_path(self.test_dir, "B02")
        self.assertIsNotNone(found_path)
        self.assertEqual(found_path, self.band_file)

    def test_find_band_path_not_found(self):
        # Test if _find_band_path returns None if the band file is not found
        found_path = _find_band_path(self.test_dir, "B03")
        self.assertIsNone(found_path)

if __name__ == '__main__':
    unittest.main()