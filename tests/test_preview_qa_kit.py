from pathlib import Path
from identity_worker.preview import _save_generated_image

class FakeImageModule:
    class Image:
        pass
    @staticmethod
    def fromarray(value):
        return value

class FakeFrame(FakeImageModule.Image):
    def convert(self, mode): return self
    def save(self, path, format=None, optimize=None): Path(path).write_bytes(b'png')


def test_qa_image_output_is_materialized(tmp_path):
    output=_save_generated_image([FakeFrame()], tmp_path/'qa.png', FakeImageModule)
    assert output.read_bytes()==b'png'
