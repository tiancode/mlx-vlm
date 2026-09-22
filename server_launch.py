#!/usr/bin/env python3
"""安装 GLM 运行时补丁后启动 mlx_vlm.server。

APC 在驱逐前限制 prefill 预留，快照可剥离 GLM 的派生投影缓存。
另安装稀疏注意力、请求清理和取消补丁；适配 mlx-vlm 0.7.1。
"""

import os

import mlx_vlm.apc as _apc
from mlx_vlm.apc import APCManager

_FRACTION = float(os.environ.get("APC_RESERVE_FRACTION", "0.25"))
if not 0 < _FRACTION <= 1:
    raise ValueError("APC_RESERVE_FRACTION must be in (0, 1]")
_original_observe = APCManager._observe_cache_size


def _cap(self) -> None:
    ceiling = int(getattr(self, "memory_max_bytes", 0) * _FRACTION)
    if ceiling > 0 and getattr(self, "_prefill_reserve_bytes", 0) > ceiling:
        self._prefill_reserve_bytes = ceiling


def _observe_cache_size(self, size, token_count):
    _original_observe(self, size, token_count)
    _cap(self)


def _prepare_prefill(self, token_count):
    # 必须在 _make_room 前封顶，否则上游调用先把已有快照全驱逐，事后无法恢复。
    if self.disk is not None:
        self.disk.flush()
    self._prefill_tokens = max(0, token_count)
    self._prefill_reserve_bytes = int(
        2 * self._prefill_tokens * self._bytes_per_token
    )
    _cap(self)
    self._make_room()


APCManager._observe_cache_size = _observe_cache_size
APCManager.prepare_prefill = _prepare_prefill


# GLM 的 CacheList 中，多头 KV 是可由压缩潜变量重算的 projected 缓存。
_STRIP_DERIVED = os.environ.get("APC_STRIP_DERIVED", "1") not in ("0", "false", "no")
_DISK_MIN_TOKENS = int(os.environ.get("APC_DISK_MIN_TOKENS", "8192"))


def _strip_entry(entry):
    """把 CacheList 里分头的 projected_cache 换成空 KVCache；其余原样返回。"""
    caches = getattr(entry, "caches", None)
    if not caches:
        return entry
    from mlx_vlm.models.cache import CacheList, KVCache

    if not isinstance(entry, CacheList):
        return entry
    out = list(caches)
    hit = False
    for i, child in enumerate(out):
        if type(child) is not KVCache:
            continue
        keys = getattr(child, "keys", None)
        # 依赖 GLM 当前结构：多头 projected 与单头潜变量 / 索引分开。
        if keys is None or keys.ndim != 4 or keys.shape[1] <= 1:
            continue
        if int(getattr(child, "offset", 0) or 0) <= 0:
            continue
        out[i] = KVCache()
        hit = True
    # 不改动传进来的对象本身：重新包一层，未命中的子项仍是同一批引用
    return CacheList(*out) if hit else entry


def _strip_derived(prompt_cache):
    if not _STRIP_DERIVED or not prompt_cache:
        return prompt_cache
    return [_strip_entry(c) for c in prompt_cache]


if _STRIP_DERIVED:
    _original_clone = _apc._clone_prompt_cache_for_apc

    def _clone_prompt_cache_for_apc(prompt_cache, **kwargs):
        # 先剥离再克隆，避免复制无需持久化的投影张量。
        return _original_clone(_strip_derived(prompt_cache), **kwargs)

    # apc.py 内部按模块全局名调用，替换模块属性即可生效
    _apc._clone_prompt_cache_for_apc = _clone_prompt_cache_for_apc

    _original_store_exact = APCManager.store_exact_cache

    def _store_exact_cache(self, token_ids, prompt_cache, *, extra_hash=0):
        # 入场预算也必须按真正存储的缓存算，不能把即将丢弃的 projected 算进去。
        return _original_store_exact(
            self, token_ids, _strip_derived(prompt_cache), extra_hash=extra_hash
        )

    APCManager.store_exact_cache = _store_exact_cache

# 磁盘准入阈值独立于剥离开关；同步落盘路径也经过此入口。
_original_save_exact = _apc.DiskBlockStore.save_exact_cache


def _save_exact_cache(self, cache_hash, token_ids, extra_hash, prompt_cache, **kw):
    if _DISK_MIN_TOKENS > 0 and len(token_ids) < _DISK_MIN_TOKENS:
        return False
    return _original_save_exact(
        self, cache_hash, token_ids, extra_hash, _strip_derived(prompt_cache), **kw
    )


_apc.DiskBlockStore.save_exact_cache = _save_exact_cache

# BatchKVCache.merge() 支持全部为空；extract() 也必须支持此状态。
# 独立于 APC_STRIP_DERIVED 开关。单边缺失不属于空缓存，保留上游异常。
from mlx_vlm.models.cache import BatchKVCache, KVCache as _KVCache

_original_extract = BatchKVCache.extract


def _extract(self, idx):
    if self.keys is None and self.values is None:
        return _KVCache()
    return _original_extract(self, idx)


BatchKVCache.extract = _extract

from long_context_patch import install, install_apc_limit

install()
install_apc_limit(APCManager)

from request_cleanup_patch import install as install_request_cleanup

install_request_cleanup()


if __name__ == "__main__":
    from request_lifecycle import install_backend

    install_backend()
    from mlx_vlm.server.cli import main

    main()
