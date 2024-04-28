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

## https://new.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/

https://github.com/LostRuins/koboldcpp

Therefore, consider this post a dual-purpose evaluation: firstly, an in-depth assessment of Llama 3 Instruct's capabilities, and secondly, a comprehensive comparison of its HF, GGUF, and EXL2 formats across various quantization levels. In total, I have rigorously tested 20 individual model versions, working on this almost non-stop since Llama 3's release.

Read on if you want to know how Llama 3 performs in my series of tests, and to find out which format and quantization will give you the best results.

Models (and quants) tested
MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, IQ4_XS, Q3_K_L, Q3_K_M, Q3_K_S, IQ3_XS, IQ2_XS, Q2_K, IQ1_M, IQ1_S

NousResearch/Meta-Llama-3-70B-Instruct-GGUF Q5_K_M

meta-llama/Meta-Llama-3-8B-Instruct HF (unquantized)

turboderp/Llama-3-70B-Instruct-exl2 5.0bpw (UPDATE 2024-04-24!), 4.5bpw, 4.0bpw

turboderp/Llama-3-8B-Instruct-exl2 6.0bpw

UPDATE 2024-04-24: casperhansen/llama-3-70b-instruct-awq AWQ (4-bit)



## https://www.reddit.com/r/LocalLLaMA/comments/1cdxjax/i_created_a_new_benchmark_to_specifically_test/
### https://new.reddit.com/r/LocalLLaMA/comments/1b5uv86/perplexity_is_not_a_good_measurement_of_how_well/
![image](https://github.com/qianxinchun/Inquiry-Odyssey-LLM-Exploration/assets/7309139/65ae54a7-2404-40ad-a185-59bfdf6a08bc)
Then exllama2 dropped and that was amazing because we got variable bit quantization options and it was fast as hell.
In general, the sentiment I've seen expressed is that you can quantize down to around 4bit (less these days) while still having around the same perplexity as the fp16 version of the model. This is true, but what I want to make absolutely clear is that perplexity is a pretty crap metric for how well a model is able to perform the tasks given to it. To me it seems that a low perplexity just means that the model is able to produce coherent, readable sentences that are at least somewhat related to the prompt. It says nothing about whether its output actually makes sense given the context of conversation or whether it was able to correctly reason and draw conclusions from information given to it.
Sure, the heavily quantized versions can write passable prose and for uncomplicated requests will generally perform fine, but their logical reasoning and recall abilities are complete trash compared to the higher precision versions.
At the end of the day I think we need to figure out better metrics than perplexity for evaluating quantization methods. It's clear that perplexity says almost nothing about how usable a model is for any non-trivial task.

https://github.com/jd-3d/MPA_Bench/tree/main

"How Good Are Low-bit Quantized LLAMA3 Models? An Empirical Study" https://arxiv.org/pdf/2404.14047

In Exllamav2 they found the 4bit cache outperformed the 8bit cache for inference. This is mysterious stuff, where we need better empirical tests.
That's because of the way that 8 cache was quantized, turboderp talked about it. 8-bit cache was quantized in a very rough manner, basically cutting off the last 8 bits of the value instead of properly quantizing it. It's no mystery at all.

llama.cpp's tokenization is not fixed yet

PR: https://github.com/ggerganov/llama.cpp/pull/6920

issue: https://github.com/ggerganov/llama.cpp/issues/6914

The issue specifically calls out that the multi-digits tokenization is wrong. You'll have to wait until it's fixed.

## https://www.reddit.com/r/LocalLLaMA/comments/16pz63a/exllamav2_quantization_colab_notebook/
EXL2 quantizing is very resource efficient. 7b models take about 8g of VRAM, even a 70b takes less than 24gb and can be done on a 3090. Try it yourself on google colab. (maybe start smaller than 70b, those take a while)

https://colab.research.google.com/drive/1Cbb8nrwUxoxAbsIu1LLotsk2W52nj0Py?usp=sharing

## k-quants #1684
https://github.com/ggerganov/llama.cpp/pull/1684

![image](https://github.com/qianxinchun/Inquiry-Odyssey-LLM-Exploration/assets/7309139/a0e9b146-ff00-483b-91ea-2dc42fd5ea57)

![image](https://github.com/qianxinchun/Inquiry-Odyssey-LLM-Exploration/assets/7309139/6a55b8ba-94b3-42bd-8c4e-6ad541c03057)

a 2-bit quantized 65B model is very similar to an 8-bit quantized 30B (in both RAM needs and quality)

an 8-bit quantized 7B model is of very similar quality as a 2-bit quantized 13B model

You probably want 3 or 4 or 5 bit quantization, if this is anything like LLAMA2. From 3 bit to 2, your quality goes down a lot while RAM requirements only go down a little. From 5 bit to 6 your quality only goes up a little.


