#LRUCacheUnhashable
#own implementation of LRU cache to handle unhashable lists

from collections import OrderedDict
from modules.config import VERBOSE, CACHE_MAX_SIZE
#import torch

class LRUCacheUnhashable:
    """
    A decorator class that caches results for functions with unhashable arguments.
    Uses an OrderedDict to implement a simple LRU eviction policy.
    """
    def __init__(self, 
                 orig_func=None, 
                 maxsize=CACHE_MAX_SIZE
                 ):
        """initialisation"""
        self.orig_func = orig_func
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, bit_string_list):
        """call wrapper"""
        key = self.list_to_bit_string(bit_string_list)
        if key in self.cache:
            self.cache_hits += 1
            if VERBOSE:
                print(f'Reading cache with key = {key}')
                print(f'Cache is now {self.cache}')
            self.cache.move_to_end(key)
            result = self.cache[key]
        else:
            self.cache_misses += 1
            result = self.orig_func(bit_string_list)
            self.cache[key] = result # store the result in the cache
            if VERBOSE:
                print(f'Updating cache with key = {key}')
            if len(self.cache) > self.maxsize:
                item = self.cache.popitem(last=False)
                if VERBOSE:
                    print(f'Evicting item {item} from cache')
                    print(f'Cache is now {self.cache}')
        return result

    @staticmethod
    def list_to_bit_string(bit_string_input):
        """convert list in format [0,1] to bit string eg '01'"""
        if isinstance(bit_string_input, list):
            bit_string_list = bit_string_input
        else:
            raise Exception(f'{bit_string_list} is not a list')
        return ''.join(map(str, bit_string_list))

    def print_cache(self):
        """print cache"""
        print(f'cache = {self.cache}')

    def print_cache_stats(self):
        """print cache stats"""
        print(f'Items in cache = {len(self.cache)}')
        print(f'cache_hit = {self.cache_hits}')
        print(f'cache_miss = {self.cache_misses}')
        if self.cache_hits + self.cache_misses == 0:
            print(f'The cache is empty - no stats available')
        else:
            self.cache_hit_rate = self.cache_hits/(self.cache_hits+self.cache_misses)
            print(f'cache_hit_rate = {self.cache_hit_rate:.3f}')

    def report_cache_stats(self):
        """reports cache stats"""
        items = len(self.cache)
        hits = self.cache_hits
        misses = self.cache_misses
        return items, hits, misses

    def clear_cache(self):
        """clear cache"""
        self.cache = OrderedDict()
        self.cache_hits, self.cache_misses = 0, 0