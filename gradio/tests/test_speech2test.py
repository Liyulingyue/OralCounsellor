import unittest
from ..utils.asr import speech2text, create_asr_model

asr_model, output_ir = create_asr_model()

class TestExecutor(unittest.TestCase):
    def hello_world(self):
        audio_test = speech2text("./mp3/helllo_world.mp3", asr_model, output_ir)
        self.assertEqual(audio_test, "Hello world")

if __name__ == '__main__':
    unittest.main()