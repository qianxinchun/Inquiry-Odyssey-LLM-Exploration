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

### RING Attention explained: 1 Mio Context Length https://www.youtube.com/watch?v=jTJcP8iyoOM
https://medium.com/@tanuj22july/breaking-the-boundaries-understanding-context-window-limitations-and-the-idea-of-ring-attention-170e522d44b2
https://github.com/lucidrains/ring-attention-pytorch
https://github.com/lhao499/RingAttention
To put the memory demand in perspective, consider processing 100 million tokens. Even with a batch size of just one, a modest Transformer model with a hidden size of 1024 would require over 1000GB of memory. This requirement far exceeds the capacity of contemporary GPUs and TPUs, which typically offer less than 100GB of high-bandwidth memory (HBM). The stark contrast between the memory demands of large context processing and the available hardware capabilities underscores the critical nature of this challenge.
Ring Attention ingeniously sidesteps this issue by breaking down the input text into smaller, manageable blocks. Each block is processed on different devices arranged in a ring-like structure, allowing for parallel processing. Here’s the clever part: as each device finishes with its block, it passes on crucial information to the next device in the ring, ensuring a continuous flow of context without overloading any single device.
The complexity of computing attention remains quadratic. However, the innovation of Ring Attention lies not in reducing the complexity per se but in enabling the processing of sequences that are much longer than what traditional models can handle by distributing the computation across multiple devices. The quadratic complexity is tackled within manageable blocks, and the ring topology ensures efficient aggregation of information across the entire sequence.

### WORLD MODEL ON MILLION-LENGTH VIDEO AND LANGUAGE WITH BLOCKWISE RINGATTENTION
https://arxiv.org/pdf/2402.08268

![image](https://github.com/qianxinchun/Inquiry-Odyssey-LLM-Exploration/assets/7309139/e185f010-deae-4665-bc68-6b3eb3daf174)


![image](https://github.com/qianxinchun/Inquiry-Odyssey-LLM-Exploration/assets/7309139/a5ad6217-bcc4-4072-95ec-14c1060cc387)

## https://www.reddit.com/r/singularity/comments/1cd9xpm/llama_3_now_with_160k_context/

## https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/
perplexity: more statistics, added documentation #6936 https://github.com/ggerganov/llama.cpp/pull/6936 
