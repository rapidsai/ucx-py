import pytest

import ucp


@pytest.mark.parametrize("blocking_progress_mode", [True, False])
def test_fence(blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)
    # this should always succeed
    ucp.fence()
    ucp.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_flush(blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)

    await ucp.flush()
    ucp.reset()
