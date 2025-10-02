import unittest
from detector import detectar_rostros  # aun no existe

class TestDeteccionRostros(unittest.TestCase):
    def test_imagen_no_existe(self):
        with self.assertRaises(FileNotFoundError):
            detectar_rostros("foto_2.jpg")

if __name__ == "__main__":
    unittest.main()
