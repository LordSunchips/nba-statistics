import unittest

from fantasy_basketball.player import Player

class TestPlayer(unittest.TestCase):
    
    def test_steph_curry(self):
        steph = Player('Stephen Curry')
        self.assertEqual(steph.full_name, 'Stephen Curry')
        self.assertEqual(steph.first_name, 'Stephen')
        self.assertEqual(steph.last_name, 'Curry')
        self.assertEqual(steph.positions, ['PG'])
        self.assertEqual(steph.team, 'GSW')
        self.assertEqual(steph.sleeper_data['birth_date'], '1988-03-14')
        
    def test_nikola_jokic(self):
        jokic = Player('Nikola Jokić')
        self.assertEqual(jokic.full_name, 'Nikola Jokić')
        self.assertEqual(jokic.first_name, 'Nikola')
        self.assertEqual(jokic.last_name, 'Jokić')
        self.assertEqual(jokic.positions, ['C'])
        self.assertEqual(jokic.team, 'DEN')
        self.assertEqual(jokic.sleeper_data['birth_date'], '1995-02-19')