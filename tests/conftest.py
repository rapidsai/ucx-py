import asyncio
import os

import pytest

import ucp

# Prevent calls such as `cudf = pytest.importorskip("cudf")` from initializing
# a CUDA context. Such calls may cause tests that must initialize the CUDA
# context on the appropriate device to fail.
# For example, without `RAPIDS_NO_INITIALIZE=True`, `test_benchmark_cluster`
# will succeed if running alone, but fails when all tests are run in batch.
os.environ["RAPIDS_NO_INITIALIZE"] = "True"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print(msg)


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.fixture()
def event_loop(scope="session"):
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    ucp.reset()
    yield loop
    ucp.reset()
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
