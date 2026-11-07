(EngineCore pid=19097) INFO 11-05 08:50:28 [kv_cache_utils.py:1147] CPU KV cache size: 9,728 tokens, Maximum concurrency for 4,096 tokens per request: 2.38x


Priming1 starts prefill. Reuse1 reaches the worker and checks the prefix cache.Priming1 has not finished inserting the relevant KV blocks. Reuse1 gets zero cached tokens.

block size is 64