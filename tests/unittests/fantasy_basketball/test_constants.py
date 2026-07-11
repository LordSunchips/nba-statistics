import unittest
from pathlib import Path

from fantasy_basketball.constants import Constants

class TestConstants(unittest.TestCase):
    def test_project_root(self):
        expected_project_root = str(Path(__file__).parent.parent.parent.parent)
        self.assertEqual(Constants.PROJECT_ROOT, expected_project_root)
        
        # Assert that src/fantasy_basketball is a subdirectory of PROJECT_ROOT
        self.assertTrue(Path(Constants.PROJECT_ROOT).joinpath('src', 'fantasy_basketball').is_dir())

    def test_data_cache_dir(self):
        expected_data_cache_dir = str(Path(Constants.PROJECT_ROOT) / 'data_cache')
        self.assertEqual(Constants.DATA_CACHE_DIR, expected_data_cache_dir)