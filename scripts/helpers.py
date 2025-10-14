import io
import sys
import json
import time
import types
import random
import string
import asyncio
import logging
import itertools
import collections.abc

from typing import Any, Callable, Iterable, Generator
from contextlib import nullcontext

import aiohttp
import msgspec

def readf(path: str, mode: str = 'r', encoding: str = 'utf-8') -> str:
    """Read a file."""
    
    with open(path, mode, encoding = encoding if 'b' not in mode else None) as file:
        return file.read()

def savef(path_or_buffer: str | io.BufferedIOBase, data: str | bytes, mode: str = 'w', encoding: str = 'utf-8') -> None:
    if isinstance(data, bytes) and 'b' not in mode:
        logging.warning("You gave `savef()` bytes to write but didn't specify to write in binary mode. We're going to write in binary mode.")
        mode = 'wb'
    
    if isinstance(path_or_buffer, str):
        with open(path_or_buffer, mode, encoding = encoding if 'b' not in mode else None) as f:
            f.write(data)
    
    elif isinstance(path_or_buffer, io.BufferedIOBase):
        path_or_buffer.write(data)
    
    else:
        raise ValueError("You must provide a path or a buffer to write to.")

def batch_generator(iterable: Iterable, batch_size: int) -> Generator[list, None, None]:
    """Generate batches of the specified size from the provided iterable."""
    
    iterator = iter(iterable)
    
    for first in iterator:
        yield list(itertools.chain([first], itertools.islice(iterator, batch_size - 1)))

def randomly_sample(data: list, sample_size: int, seed: int = None) -> list:
    """Randomly sample up to the specified number of elements from the provided data."""

    # Set the seed if one was provided.
    if seed:
        random.seed(seed)

    # Randomly shuffle the data.
    random.shuffle(data)

    # Return up to `sample_size` elements from the data.
    data = data[:sample_size]
    
    # Unset the seed if one was provided.
    if seed:
        random.seed()
    
    return data


def randomly_select(data: list, seed: int = None) -> str:
    """Randomly select an element from the provided data."""

    # Set the seed if one was provided.
    if seed:
        random.seed(seed)

    # Randomly select an element from the data.
    data = random.choice(data)
    
    # Unset the seed if one was provided.
    if seed:
        random.seed()
    
    return data

def randomly_shuffle(data: list, seed: int = None) -> list:
    """Randomly shuffle the provided data."""
    
    # Set the seed if one was provided.
    if seed:
        random.seed(seed)

    # Randomly shuffle the data.
    random.shuffle(data)

    # Unset the seed if one was provided.
    if seed:
        random.seed()

    return data

def print_header(text: str, char_or_level: str | int = '-', width: int = 100) -> None:
    """Print a header."""
    
    # Identify the character to use for the header.
    if isinstance(char_or_level, int):
        char = ['#', '=', '~', '-'][char_or_level - 1]
    
    else:
        char = char_or_level
    
    # Compute the space necessary for the header.
    text_length = len(text) + 2
    
    # Raise an error if the space needed is more than the specified width.
    if text_length > width:
        raise ValueError('The space necessary for the header is larger than the specified width.')
    
    # Calculate the number of characters needed to affix to either size of the text.
    affix_length = (width - text_length) // 2
    
    # Construct the affixed text.
    affixed_text = f"{char * affix_length} {text} {char * affix_length}"
    
    # Add an extra character to the end of the affixed text if the total width is odd.
    if len(affixed_text) < width:
        affixed_text += char
    
    # Print the header.
    print(affixed_text)

def count_lines(path: str) -> int:
    """Count the number of lines in a file."""
    
    with open(path, 'rb') as file:
        return sum(1 for _ in file)

def flatten(l: list[list]) -> list:
    """Flatten a list of lists."""
    
    return [item for sublist in l for item in sublist]

def save_json(path: str, content: Any, encoder: Callable[[Any], bytes] = msgspec.json.Encoder().encode) -> None:
    """Save content as a json file."""
    
    with open(path, 'wb') as writer:
        writer.write(encoder(content))

def load_json(path: str, decoder: Callable[[bytes], Any] = msgspec.json.Decoder().decode) -> Any:
    """Load a json file."""
    
    if not isinstance(decoder, (types.FunctionType, types.BuiltinFunctionType)):
        decoder = msgspec.json.Decoder(decoder).decode
    
    with open(path, 'rb') as reader:
        return decoder(reader.read())

def load_jsonl(path: str, decoder: Callable[[bytes], Any] = msgspec.json.Decoder().decode) -> list:
    """Load a jsonl file."""

    if not isinstance(decoder, (types.FunctionType, types.BuiltinFunctionType)):
        decoder = msgspec.json.Decoder(decoder).decode
    
    with open(path, 'rb') as file:
        return [decoder(json) for json in file]

def stream_jsonl(path: str, decoder: Callable[[bytes], Any] = msgspec.json.Decoder().decode) -> Generator[Any, None, None]:
    """Stream a jsonl file."""
    
    if not isinstance(decoder, (types.FunctionType, types.BuiltinFunctionType)):
        decoder = msgspec.json.Decoder(decoder).decode
    
    if hasattr(decoder, 'decode'):
        decoder = decoder.decode
    
    with open(path, 'rb') as file:
        for line in file:
            yield decoder(line)

def save_jsonl(path: str, content: list, encoder: Callable[[Any], bytes] = msgspec.json.Encoder().encode) -> None:
    """Save a list of objects as a jsonl file."""
    
    with open(path, 'wb') as file:
        for entry in content:
            file.write(encoder(entry))
            file.write(b'\n')

def comma(number: int) -> str:
    """Add commas to a number."""
    
    return f'{number:,}'

def dedup(l: list) -> list:
    """Deduplicate a list."""
    
    try:
        nl = list(set(l))
    
    except TypeError:
        nl = []
        
        for item in l:
            if item not in nl:
                nl.append(item)
    
    return nl

def level(l: list[list]) -> list:
    """Completely level a list of infinitely nested lists."""
    
    return list(itertools.chain.from_iterable(itertools.repeat(sublist, 1) if not isinstance(sublist, list) else level(sublist) for sublist in l))

def depunc(text: str) -> str:
    """Remove punctuation from text."""
    
    return text.translate(str.maketrans('', '', string.punctuation))

async def fetch(url: str, session: aiohttp.ClientSession | None = None, semaphore: asyncio.Semaphore | None = None, method: str = 'GET', data: dict | None = None, headers: dict | None = None, cookies: dict | None = None) -> bytes:
    """Asynchronously fetch a URL, automatically implementing exponential backoff with jitter."""
    
    close_session = False
    
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    
    if semaphore is None:
        semaphore = nullcontext()
    
    attempt = 0
    elapsed = 0
    
    while True:
        try:
            async with semaphore:
                async with session.request(method, url, data = data, headers = headers, cookies = cookies) as response:
                    response = await response.read()
            
            if close_session:
                await session.close()
                        
            return response
        
        # Implement exponential backoff with jitter.
        except aiohttp.ClientError as e:
            if elapsed > 30 * 60:
                raise e
            
            attempt += 1
            
            wait = 1.25 ** (attempt / 2) # We divide by 2 so that `wait + jitter` is always <= `1.5 ** attempt`.
            
            # Set our jitter to a random number between 0 and `wait`.
            jitter = random.uniform(0, wait)
            
            wait = wait + jitter
            
            # If `wait` is greater than `self.max_wait`, set `wait` to `2.5 * 60`.
            wait = min(wait, 2.5 * 60) + random.uniform(0, 1)
            
            # Wait for `wait` seconds.
            await asyncio.sleep(wait)
            
            elapsed += wait

def pretty(data: Any) -> str:
    """Pretty print data."""
    
    json_data = msgspec.json.Encoder().encode(data)
    data = msgspec.json.decode(json_data)
    
    return json.dumps(data, indent = 4)

# Adapted by Umar Butler from Andrei Lapets' `sizeof.sizeof()` function, licensed under the MIT License, available at https://github.com/lapets/sizeof/blob/c25fff4259e1d99c5000d813a2f91eaa850464ab/src/sizeof/sizeof.py.
def getsizeof(
    obj: Any,
    _counted: set[int] = None,
) -> int:
    if isinstance(obj, (int, float, complex, bool, str, bytes, bytearray, memoryview)):
        return sys.getsizeof(obj)
    
    _counted = _counted or set()
    
    if id(obj) in _counted:
        return 0
    
    if isinstance(obj, collections.abc.Mapping):
        _counted.add(id(obj))
        
        return sys.getsizeof(obj) + sum([getsizeof(k, _counted) + getsizeof(v, _counted) for k, v in obj.items()])
    
    if isinstance(obj, collections.abc.Container):
        _counted.add(id(obj))
        
        return sys.getsizeof(obj) + sum([getsizeof(x, _counted) for x in obj])
    
    raise ValueError(f'"{type(obj)}" is not a type supported by `getsizeof()`.')

def timeit(
    func_: Callable,
    *args: Any,
    runs_: int = 1,
    **kwargs: Any,
) -> dict[str, float]:
    """Time a function."""
    
    times = []
    
    for _ in range(runs_):
        start = time.perf_counter()
        func_(*args, **kwargs)
        times.append(time.perf_counter() - start)
    
    total = sum(times)
    average = total / runs_
    minimum = min(times)
    maximum = max(times)
    
    return {
        'total': total,
        'average': average,
        'minimum': minimum,
        'maximum': maximum,
    }