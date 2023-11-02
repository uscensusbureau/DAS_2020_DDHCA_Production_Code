import re
import pytest
import ctools.timer as timer


def test_timer_message_no_percentage():
    test_timer = timer.Timer(message="Test:")
    test_message = test_timer.message
    print(f"test_timer_message_no_percentage: {test_message}")
    assert test_message == "Test: %f seconds"


def test_timer_message_with_percentage():
    test_timer = timer.Timer(message="Test: %f %f seconds")
    test_message = test_timer.message
    print(f"test_timer_message_with_percentage: {test_message}")
    assert test_message == "Test: %f %f seconds"


def test_timer_default_message():
    test_timer = timer.Timer()
    test_message = test_timer.message
    print(f"test_timer_default_message: {test_message}")
    assert test_message == "Elapsed time: %f seconds"


def test_timer_interval(capfd):
    test_timer = timer.Timer()
    test_timer.__enter__()
    test_timer.__exit__()
    out, err = capfd.readouterr()
    print(f"test_timer_interval: {err}")
    assert f"{err}".strip().startswith("Elapsed time: ")
    assert f"{err}".strip().endswith("seconds")
    assert "%f" not in f"{err}".strip()


def test_timer_interval_no_message(capfd):
    test_timer = timer.Timer()
    test_timer.__enter__()
    test_timer.message = None
    test_timer.__exit__()
    out, err = capfd.readouterr()
    print(f"test_timer_interval_no_message: {err}")
    assert f"{err}".strip() == ""


def test_timer_interval_no_notifier(capfd):
    test_timer = timer.Timer()
    test_timer.__enter__()
    test_timer.notifier = None
    test_timer.__exit__()
    out, err = capfd.readouterr()
    print(f"test_timer_interval_no_notifier: {err}")
    assert f"{err}".strip() == ""


def test_print_stderr(capfd):
    timer.print_stderr("Test Message")
    out, err = capfd.readouterr()
    print(f"test_print_stderr: {err}")
    assert f"{err}".strip() == "Test Message"
