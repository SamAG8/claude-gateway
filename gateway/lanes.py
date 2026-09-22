"""Keyed concurrency lanes: how many invocations of a kind may run at once.

Extracted from the engine because both engines need the same rule and neither
owns it. A lane is a named semaphore plus one policy: a request may wait at most
``QUEUE_WAIT_MAX`` seconds for a slot, after which the gateway fails fast with a
503 the client can react to instead of letting it burn its whole timeout budget
in an invisible queue.

Three lanes, and the split is the point:

  fast   CLI, latency-sensitive tier (haiku). Its own capacity so an interactive
         call never queues behind a long extraction job.
  heavy  CLI, everything else (opus/sonnet, up to the 300s budget).
  http   The OpenRouter engine. Never shares with the CLI lanes: a subprocess
         costs CPU and RAM on this host, an HTTP stream costs a socket, so one
         capacity cannot describe both — and a GLM turn queueing behind CLI work
         would recreate the very throttling the second engine exists to relieve.

The cap on ``http`` is not there to protect this host. It is the cheapest breaker
against a retry storm tripping the upstream's own rate limit and cascading 429s
to everyone.
"""
import asyncio

from . import config

# Created lazily so each binds to the running event loop.
_semaphores: dict[str, asyncio.Semaphore] = {}

_CAPACITY = {
    "fast": lambda: config.MAX_CONCURRENT_FAST,
    "heavy": lambda: config.MAX_CONCURRENT,
    "http": lambda: config.MAX_CONCURRENT_HTTP,
}


class Saturated(Exception):
    """No slot became free within QUEUE_WAIT_MAX. Carries the wait for logging."""

    def __init__(self, lane: str, waited_s: float):
        super().__init__(f"lane {lane} saturated after {waited_s:.1f}s")
        self.lane = lane
        self.waited_s = waited_s
        self.queue_wait_ms = int(waited_s * 1000)


def get_semaphore(lane: str) -> asyncio.Semaphore:
    sem = _semaphores.get(lane)
    if sem is None:
        cap = _CAPACITY.get(lane, _CAPACITY["heavy"])()
        sem = asyncio.Semaphore(cap)
        _semaphores[lane] = sem
    return sem


async def acquire(lane: str) -> int:
    """Take a slot in ``lane``; return the queue wait in ms.

    Raises ``Saturated`` if no slot came free in time. The caller owns the
    matching ``release`` in a ``finally`` — deliberately explicit rather than a
    context manager, because both call sites are async generators whose slot must
    survive every yield and be freed on GeneratorExit.
    """
    loop = asyncio.get_event_loop()
    enqueue_t = loop.time()
    try:
        await asyncio.wait_for(get_semaphore(lane).acquire(),
                               timeout=config.QUEUE_WAIT_MAX)
    except asyncio.TimeoutError:
        raise Saturated(lane, loop.time() - enqueue_t) from None
    return int((loop.time() - enqueue_t) * 1000)


def release(lane: str) -> None:
    get_semaphore(lane).release()
