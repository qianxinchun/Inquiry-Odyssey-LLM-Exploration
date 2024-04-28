# Inquiry-Odyssey-LLM-Exploration

## Inquiry Odyssey: Unveiling the Depths of Large Language Models

1. Why can't the gpt4-turbo guarantee prediction stability even when top_p=0.0? (reference: [https://zhuanlan.zhihu.com/p/688703803](https://zhuanlan.zhihu.com/p/688703803))
2. In the Mistral 7B v0.2 Base Model, Rope Theta is set to 1e6. Please analyze what effect this setting would have?
3. Why should we do online RLHF? (reference: https://zhuanlan.zhihu.com/p/688806682)

## https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k
### Approach:

meta-llama/Meta-Llama-3-8B-Instruct as the base

NTK-aware interpolation [1] to initialize an optimal schedule for RoPE theta, followed by a new data-driven RoPE theta optimization technique

Progressive training on increasing context lengths similar to the Large World Model [2] (See details below)

### Infra:

We build on top of the EasyContext Blockwise RingAttention library [3] to scalably and efficiently train on contexts up to 262144 tokens on Crusoe Energy high performance L40S cluster.

### References

[1] Peng, Bowen, et al. "Yarn: Efficient context window extension of large language models." arXiv preprint arXiv:2309.00071 (2023).

[2] Liu, Hao, et al. "World Model on Million-Length Video And Language With RingAttention." arXiv preprint arXiv:2402.08268 (2024).

[3] https://github.com/jzhang38/EasyContext

### Extending the RoPE https://blog.eleuther.ai/yarn/
https://blog.eleuther.ai/rotary-embeddings/
https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

