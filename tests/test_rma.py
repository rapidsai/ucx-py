import pytest

import ucp


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_fence(blocking_progress_mode):
    # Test needs to be async here to ensure progress tasks are cleared
    # and avoid warnings.

    ucp.init(blocking_progress_mode=blocking_progress_mode)
    # this should always succeed
    ucp.fence()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_flush(blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)

    await ucp.flush()
