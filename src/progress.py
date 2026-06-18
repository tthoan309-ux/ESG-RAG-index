from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def progress(iterable: Iterable[T], *args, **kwargs) -> Iterator[T]:
    try:
        from tqdm import tqdm

        yield from tqdm(iterable, *args, **kwargs)
    except ImportError:
        yield from iterable
