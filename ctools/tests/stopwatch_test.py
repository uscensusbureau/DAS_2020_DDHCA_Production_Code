import pytest
import ctools.stopwatch as stopwatch


@pytest.fixture()
def test_stopwatch():
    test_stopwatch = stopwatch.stopwatch()
    yield test_stopwatch


def test_stopwatch_init(test_stopwatch):
    assert test_stopwatch.running is False


def test_stopwatch_start(test_stopwatch):
    test_stopwatch.start()
    assert test_stopwatch.running is True
    assert isinstance(test_stopwatch.t0, float)


def test_stopwatch_stop(test_stopwatch):
    test_stopwatch.stop()
    assert test_stopwatch.running is False
    assert isinstance(test_stopwatch.t1, float)


def test_stopwatch_elapsed(test_stopwatch):
    test_stopwatch.start()
    test_stopwatch.stop()
    elapsed_time = test_stopwatch.t1 - test_stopwatch.t0
    assert test_stopwatch.elapsed() == elapsed_time


def test_stopwatch_elapsed_still_running(test_stopwatch):
    test_stopwatch.start()
    assert isinstance(test_stopwatch.elapsed(), float)
