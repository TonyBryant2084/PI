import jax
import jax.numpy as jnp

from openpi.models.pi0_config import Pi0Config

def main():

    # Jax 当前版本 (JAX version: 0.5.3)
    print("JAX version:", jax.__version__)

    # 获取 JAX 当前可用的所有设备，例如 CPU、GPU (Devices: [CudaDevice(id=0)])
    print("Devices:", jax.devices())

# 2. 配置 Pi0 模型
    # config = Pi0Config(
    #     dtype="bfloat16",                     # 词嵌入结果的指定类型，模型计算使用 bfloat16 精度
    #     paligemma_variant="gemma_2b",         # 使用 gemma_2b 配置的 PaliGemma 视觉语言模型
    #     action_expert_variant="gemma_300m",   # 使用 gemma_300m 配置的 Action Expert 动作模型
    #     action_dim=32,                        # 每个动作的维度为 32
    #     action_horizon=50,                    # 每次预测连续 50 个动作
    #     max_token_len=48,                     # 语言提示最多包含 48 个 token
    #     discrete_state_input=False,           # 不将状态输入作为离散 token 处理
    #     pi05=False,                           # 不使用 Pi0.5 模型配置
    #     pytorch_compile_mode="max-autotune",  # 使用 PyTorch 编译的 max-autotune 模式，最大化自动调优
    # )
    config = Pi0Config(
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
    )

    # 3. 随机初始化模型，不加载预训练权重
    #   1) 创建 JAX 随机数密钥，用于控制随机数的生成，使用整数 0 作为随机种子，便于复现实验    
    rng = jax.random.PRNGKey(0)
    #   2) 创建 PI0 模型
    model = config.create(rng)

    print("Model initialized.")

    # observation：机器人观测数据
    # batch_size = 1
    # height = 224, width = 224, channels = 3
    # state_dim = 32, seq_len = 48
    #
    # observation = Observation(
    #     # 三路 RGB 图像：[batch_size, height, width, channels]
    #     images={
    #         "base_0_rgb": ...,         # [1, 224, 224, 3]
    #         "left_wrist_0_rgb": ...,   # [1, 224, 224, 3]
    #         "right_wrist_0_rgb": ...,  # [1, 224, 224, 3]
    #     },
    #
    #     # 图像有效性掩码：[batch_size]
    #     image_masks={
    #         "base_0_rgb": ...,         # [1]
    #         "left_wrist_0_rgb": ...,   # [1]
    #         "right_wrist_0_rgb": ...,  # [1]
    #     },
    #
    #     state=...,                    # [batch_size, state_dim] = [1, 32]
    #     tokenized_prompt=...,         # [batch_size, seq_len] = [1, 48]
    #     tokenized_prompt_mask=...,    # [batch_size, seq_len] = [1, 48]
    #
    #     # π0-FAST 专用，普通 π0 通常为 None
    #     token_ar_mask=None,           # [batch_size, seq_len] 或 None
    #     token_loss_mask=None,         # [batch_size, seq_len] 或 None
    # )

    # actions：目标动作序列
    # [batch_size, action_horizon, action_dim] = [1, 50, 32]
    observation = config.fake_obs(batch_size=1)

    # 5. 创建随机动作序列
    # actions: [batch_size(1), action_horizon(50), action_dim(32)]
    actions = jax.random.normal(
        jax.random.PRNGKey(1),
        shape=(1, config.action_horizon, config.action_dim),
        dtype=jnp.float32,
    )

    print("Observation state shape:", observation.state.shape)
    print("Action shape:", actions.shape)

    # 6. 计算 Pi0 的流匹配损失
    loss_rng = jax.random.PRNGKey(2)
    loss = model.compute_loss(
        loss_rng,
        observation,
        actions,
        train=False,
    )

    # 7. 等待 GPU 计算完成，再读取结果
    loss.block_until_ready()

    print("Loss shape:", loss.shape)
    print("Loss:", loss)


if __name__ == "__main__":
    main()
