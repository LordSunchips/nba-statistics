import pandas as pd
import unittest
from unittest.mock import patch, MagicMock

from fantasy_basketball.player import Player

class TestPlayer(unittest.TestCase):
    def setUp(self):
        Player.get_sleeper_data()
    
    def test_steph_curry_from_file(self):
        steph = Player('Stephen Curry')
        self.assertEqual(steph.full_name, 'Stephen Curry')
        self.assertEqual(steph.first_name, 'Stephen')
        self.assertEqual(steph.last_name, 'Curry')
        self.assertEqual(steph.positions, ['PG'])
        self.assertEqual(steph.team, 'GSW')
        self.assertEqual(steph.sleeper_data['birth_date'], '1988-03-14')
    
    @patch('requests.get')
    @patch('pathlib.Path.exists', return_value=False)
    @patch('pandas.DataFrame.to_json', return_value=None)
    def test_nikola_jokic_from_sleeper(self, mock_to_json, mock_exists, mock_get):
        mock_sleeper_data = MagicMock()
        mock_sleeper_data.json.return_value = {'123': {
            'full_name': 'Nikola Jokić',
            'first_name': 'Nikola',
            'last_name': 'Jokić',
            'fantasy_positions': ['C'],
            'team': 'DEN',
            'birth_date': '1994-09-20'
        }}
        mock_get.return_value = mock_sleeper_data

        jokic = Player('Nikola Jokić')
        self.assertEqual(jokic.full_name, 'Nikola Jokić')
        self.assertEqual(jokic.first_name, 'Nikola')
        self.assertEqual(jokic.last_name, 'Jokić')
        self.assertEqual(jokic.positions, ['C'])
        self.assertEqual(jokic.team, 'DEN')
        self.assertEqual(jokic.sleeper_data['birth_date'], '1994-09-20')
        
    def test_nonexistent_player(self):
        with self.assertRaises(ValueError) as context:
            Player('Non Existent Player')
        self.assertIn('Player with full name (Non Existent Player) not found in active players.', str(context.exception))
        
        