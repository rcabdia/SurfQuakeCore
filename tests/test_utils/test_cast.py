import unittest
from datetime import datetime

from surfquakecore.utils import Cast


class TestCast(unittest.TestCase):

    def test_cast_to_int(self):
        self.assertEqual(Cast("1", int), 1)
        self.assertEqual(Cast(b"1", int), 1)

    def test_cast_to_float(self):
        self.assertEqual(Cast("1.1", float), 1.1)
        self.assertEqual(Cast(b"1.1", float), 1.1)

    def test_cast_to_str(self):
        self.assertEqual(Cast(1, str), "1")
        self.assertEqual(Cast(b'1', str), "1")

    def test_cast_to_datetime(self):
        date_str = "28/02/2022 02:07:59.433"
        expect_date = datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S.%f')
        self.assertEqual(Cast(date_str, datetime), expect_date)


if __name__ == '__main__':
    unittest.main()
