# import pytest and use it in the test functions
import pytest

from .._data import flatten, iter_lines, iter_transpose


@pytest.fixture
def files(tmp_path):
    contents = ["One\nTwo\nThree\n", "Aaa\nBbb\nCcc\n", "Xxx\nYyy\nZzz\n"]
    paths = []

    for n, content in enumerate(contents, start=1):
        path = (tmp_path / f"test{n}.txt").resolve()
        paths.append(path)
        path.write_text(content)

    return contents, paths


def test_iter_lines(files):
    contents, paths = files

    sources, _line_nos, lines = iter_transpose(iter_lines(paths))

    assert set(sources) == set(paths)
    assert lines == tuple(flatten(map(str.splitlines, contents)))
