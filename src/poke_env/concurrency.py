import asyncio
import atexit
import sys
from logging import CRITICAL, disable
from threading import Thread
from typing import Any, List


def __run_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def __stop_loop(loop: asyncio.AbstractEventLoop, thread: Thread):
    disable(CRITICAL)
    tasks: List[asyncio.Task[Any]] = []
    for task in asyncio.all_tasks(loop):
        task.cancel()
        tasks.append(task)

    cancelled = False
    shutdown = asyncio.run_coroutine_threadsafe(loop.shutdown_asyncgens(), loop)
    shutdown.result()

    while not cancelled:
        cancelled = True
        for task in tasks:
            if not task.done():
                cancelled = False

    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.call_soon_threadsafe(loop.close)
    # disable(CRITICAL)
    # tasks = asyncio.all_tasks(loop)
    # for task in tasks:
    #     task.cancel()

    # loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

    # loop.run_until_complete(loop.shutdown_asyncgens())

    # loop.call_soon_threadsafe(loop.stop)
    # thread.join()
    # loop.call_soon_threadsafe(loop.close)


def __clear_loop():
    __stop_loop(POKE_LOOP, _t)


async def _create_in_poke_loop_async(cls_: Any, *args: Any, **kwargs: Any) -> Any:
    return cls_(*args, **kwargs)


def create_in_poke_loop(cls_: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        # Python >= 3.7
        loop = asyncio.get_running_loop()
    except AttributeError:
        # Python < 3.7 so get_event_loop won't raise exceptions
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # asyncio.get_running_loop raised exception so no loop is running
        loop = None
    if loop == POKE_LOOP:
        return cls_(*args, **kwargs)
    else:
        return asyncio.run_coroutine_threadsafe(
            _create_in_poke_loop_async(cls_, *args, **kwargs), POKE_LOOP
        ).result()


async def handle_threaded_coroutines(coro: Any):
    if not POKE_LOOP.is_running():
        print("POKE_LOOP is not running!")
    if not _t.is_alive():
        print("_t thread is not alive!")
    task = asyncio.run_coroutine_threadsafe(coro, POKE_LOOP)
    await asyncio.wrap_future(task)
    return task.result()
    # try:
    #     result = await asyncio.wait_for(asyncio.wrap_future(task), timeout=3)
    #     # result = await asyncio.wrap_future(task)
    #     print(f"Task result: {result}")
    #     return result
    # except Exception as e:
    #     print(f"Exception in task: {e}")
    #     raise


POKE_LOOP = asyncio.new_event_loop()
py_ver = sys.version_info
_t = Thread(target=__run_loop, args=(POKE_LOOP,), daemon=True)
_t.start()
atexit.register(__clear_loop)
