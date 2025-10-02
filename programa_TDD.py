import unittest
from detector import detectar_rostros  # aún no existe

class TestDeteccionRostros(unittest.TestCase):
    def test_imagen_no_existe(self):
        with self.assertRaises(FileNotFoundError):
            detectar_rostros("foto.jpg")

if __name__ == "__main__":
    unittest.main()
