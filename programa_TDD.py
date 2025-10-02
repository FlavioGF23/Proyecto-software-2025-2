import unittest
from detector import detectar_rostros

class TestDeteccionRostros(unittest.TestCase):
    def test_imagen_no_existe(self):
        # Debe lanzar error si la imagen no existe
        with self.assertRaises(FileNotFoundError):
            detectar_rostros("no_existe.jpg")

    def test_imagen_si_existe(self):
        # Debe devolver una lista/tupla si la imagen sí existe
        faces = detectar_rostros("foto.jpg")
        self.assertIsInstance(faces, (list, tuple))
        self.assertGreaterEqual(len(faces), 0)

if __name__ == "__main__":
    unittest.main()

