# AdEMAMix Optimizer

Momentum based optimizers are central to a wide range of machine learning applications. These typically rely on an Exponential Moving Average (EMA) of gradients, which decays exponentially the present contribution of older gradients. This accounts for gradients being local linear approximations which lose their relevance as the iterate moves along the loss landscape. This work questions the use of a single EMA to accumulate past gradients and empirically demonstrates how this choice can be sub-optimal: a single EMA cannot simultaneously give a high weight to the immediate past, and a non-negligible weight to older gradients. Building on this observation, we propose AdEMAMix, a simple modification of the Adam optimizer with a mixture of two EMAs to better take advantage of past gradients. Our experiments on language modeling and image classification show -- quite surprisingly -- that gradients can stay relevant for tens of thousands of steps. They help to converge faster, and often to lower minima: e.g., a 1.3B parameter AdEMAMix LLM trained on 101B tokens performs comparably to an AdamW model trained on 197B tokens (+95%) . Moreover, our method significantly slows-down model forgetting during training. Our work motivates further exploration of different types of functions to leverage past gradients, beyond EMAs.

Results from paper:

![image](https://github.com/user-attachments/assets/1a7db615-31bc-4967-b34d-218f7f7f272a)
![image](https://github.com/user-attachments/assets/c9f3bf95-7fc8-4a5a-81c3-6f3126aa49ce)

Algorithm :

![image](https://github.com/user-attachments/assets/765e29ee-c330-496f-8130-30c37b7ed6cc)

Results on Cifar10:

![image](https://github.com/user-attachments/assets/ff6329ca-0edd-439d-9cb9-dde226e9ad77)

Reference:

Matteo Pagliardini, Pierre Ablin, David Grangier (2024). The AdEMAMix Optimizer: Better, Faster, Older
paper : https://arxiv.org/abs/2409.03137
