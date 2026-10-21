import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# 创建一个长度为 7 的动作列表，前 6 个值为 0.0，最后一个值为 -1.0
# 用于 LIBERO 仿真初始化时执行占位动作
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# 设置 LIBERO 仿真环境中相机渲染图像的分辨率为 256 × 256 像素
LIBERO_ENV_RESOLUTION = 256

# @dataclasses.dataclass：自动生成 __init__() 等方法，
# 使得可以通过 Args() 创建配置对象，并使用默认值初始化参数。
@dataclasses.dataclass
# 用于集中定义 LIBERO 评估脚本的配置参数
class Args:

    # =========================================================
    # 一、Model server parameters：模型服务器参数
    # =========================================================
    
    # 模型服务器的主机地址
    host: str = "0.0.0.0"
    # 模型服务器的端口地址
    port: int = 8000

    # resize_size：输入模型之前，图像缩放并填充后的目标尺寸。
    # 主相机图像和腕部相机图像均处理为: [224, 224, 3]
    resize_size: int = 224

    # replan_steps：每次模型推理后，连续执行的动作数量。
    # 例如模型预测 action_chunk，Shape: [50, 7]，replan_steps = 5 时，只执行前 5 个动作，然后根据最新观测重新请求模型预测。
    replan_steps: int = 5

    # ===================================================================
    # 二、LIBERO environment-specific parameters：LIBERO 仿真环境参数
    # ===================================================================

    # task_suite_name：需要评估的任务集合名称。
    # 1. libero_spatial：空间关系任务集合。
    # 2. libero_object：物体相关任务集合。
    # 3. libero_goal：目标状态任务集合。
    # 4. libero_10：包含 10 个任务的集合。
    # 5. libero_90：包含 90 个任务的集合。
    task_suite_name: str = (
        "libero_spatial"  
    )

    # num_steps_wait：正式执行任务前的等待步数。
    # 等待期间执行 LIBERO_DUMMY_ACTION，
    # 让仿真中的物体完成掉落、碰撞并趋于稳定。
    num_steps_wait: int = 10

    # num_trials_per_task：每个任务执行的 episode 数量。
    # 例如 10 个任务，每个任务执行 50 次，
    # 理想情况下总共执行 10 × 50 = 500 个 episode。
    num_trials_per_task: int = 50

    # =========================================================
    # 三、Utils：辅助配置参数
    # =========================================================

    # video_out_path：episode 回放视频的保存目录。
    # 若目录不存在，后续代码会自动创建。
    video_out_path: str = "data/libero/videos" 

    # seed：随机种子，用于提高实验的可复现性。
    # 后续用于设置 NumPy 随机数生成器和仿真环境。
    # 注意：设置随机种子不一定能保证整个系统完全确定。
    seed: int = 7

def eval_libero(
    args: Args,
) -> None:
    # ============================================================
    # 1. 设置随机种子，初始化 LIBERO 任务集
    # ============================================================

    # args.seed：随机种子，用于控制随机数生成，便于复现实验。
    np.random.seed(args.seed)

    # benchmark.get_benchmark_dict()：
    # 获取 LIBERO 支持的任务集字典。
    # 返回值：任务集名称 -> 任务集构造函数。
    benchmark_dict = benchmark.get_benchmark_dict()

    # args.task_suite_name：任务集名称，例如 "libero_object"。
    # benchmark_dict[args.task_suite_name]()：创建指定任务集实例。
    task_suite = benchmark_dict[args.task_suite_name]()

    # task_suite.n_tasks：当前任务集包含的任务数量。
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(f"Task suite: {args.task_suite_name}")

    # 创建视频输出目录。
    # parents=True：父目录不存在时一并创建。
    # exist_ok=True：目录已经存在时不报错。
    pathlib.Path(args.video_out_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 2. 根据任务集设置单个 Episode 的最大步数
    # ============================================================

    # max_steps：每个 Episode 允许执行的最大任务控制步数。
    # 不同任务集的任务复杂度不同，因此限制不同。
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # 最长训练演示约 193 步

    elif args.task_suite_name == "libero_object":
        max_steps = 280  # 最长训练演示约 254 步

    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # 最长训练演示约 270 步

    elif args.task_suite_name == "libero_10":
        max_steps = 520  # 最长训练演示约 505 步

    elif args.task_suite_name == "libero_90":
        max_steps = 400  # 最长训练演示约 373 步

    else:
        # 任务集名称不受支持时，抛出异常并终止执行。
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # ============================================================
    # 3. 创建 WebSocket 客户端，连接 π0.5 推理服务器
    # ============================================================

    # args.host：推理服务器地址。
    # args.port：推理服务器端口。
    #
    # client：WebSocket 策略客户端。
    # 后续通过 client.infer(element)：
    #   1. 将当前观测发送给策略服务器。
    #   2. 等待服务器完成模型推理。
    #   3. 接收模型预测的动作序列。
    #
    # 注意：这里仅创建客户端，不在此处加载模型权重。
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )

    # ============================================================
    # 4. 初始化整个评测过程的统计变量
    # ============================================================

    # total_episodes：所有任务累计完成的 Episode 数量。
    # total_successes：所有任务累计成功的 Episode 数量。
    total_episodes, total_successes = 0, 0

    # ============================================================
    # 5. 遍历任务集中的每一个任务
    # ============================================================

    # task_id：当前任务的索引。
    # num_tasks_in_suite：任务总数量
    # range(num_tasks_in_suite)：从 0 遍历到 num_tasks_in_suite - 1。
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        # 获取当前任务的定义。
        # task 包含当前任务对应的环境配置和任务信息。
        task = task_suite.get_task(task_id)

        # 获取当前任务预定义的初始状态集合。
        # initial_states：当前任务的初始仿真状态数据集合。
        # initial_states[episode_idx]：第 episode_idx 次实验使用的初始状态，
        # 可能包含机械臂关节状态、物体位置与姿态等信息。
        # 同一个任务可以使用多份预定义初始状态，但不一定对应不同的场景布局。
        initial_states = task_suite.get_task_init_states(task_id)

        # 根据当前任务定义创建仿真环境，并获取任务的自然语言描述。
        # env：仿真环境对象，包含机械臂、物体、摄像头等仿真对象，
        #      并提供 reset()、set_init_state()、step() 等交互接口。
        # task_description：当前任务的自然语言描述，后续作为 π0.5 的语言输入。
        # 注意：创建 env 不等于设置本次实验的具体初始状态。
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
        )

        # ========================================================
        # 6. 初始化当前任务的评测统计量
        # ========================================================

        # task_episodes：当前任务已经完成的 Episode 数量。
        # task_successes：当前任务成功的 Episode 数量。
        task_episodes, task_successes = 0, 0

        # 每个任务执行 args.num_trials_per_task 次评测。
        # args.num_trials_per_task: 每个任务执行的 episode 数量
        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task)
        ):
            logging.info(f"\nTask: {task_description}")

            # ----------------------------------------------------
            # 6.1 重置环境和动作队列
            # ----------------------------------------------------

            # 重置当前仿真环境。
            env.reset()

            # action_plan：动作队列，用于保存模型预测的待执行动作。
            # collections.deque：双端队列。
            # popleft()：取出队首动作并将其移除。
            #
            # 每次开始新的 Episode 时清空队列。
            action_plan = collections.deque()

            # 每次测评使用一份与当前 Episode 索引一一对应的预定义初始状态。
            # args.num_trials_per_task 决定当前任务的测评次数，
            # episode_idx 从 0 开始递增，每次测评依次使用 initial_states[episode_idx]。
            # 因此，第 1 次测评使用 initial_states[0]，第 2 次测评使用 initial_states[1]，
            # 以此类推；初始状态数量必须不少于测评次数，否则可能发生索引越界。
            # env.set_init_state(...)：将本次测评对应的初始状态应用到仿真环境。
            # obs：初始化后的环境观测，后续从中提取图像、机器人状态等 π0.5 推理所需的数据。
            obs = env.set_init_state(
                initial_states[episode_idx]
            )

            # ----------------------------------------------------
            # 6.2 初始化 Episode 内部变量
            # ----------------------------------------------------

            # t：当前 Episode 的仿真步数计数器。
            t = 0

            # replay_images：保存预处理后的图像，用于生成回放视频。
            replay_images = []

            logging.info(
                f"Starting episode {task_episodes + 1}..."
            )

            # 最多运行 max_steps + args.num_steps_wait 步。
            #
            # 前 args.num_steps_wait 步用于等待仿真物体稳定；
            # 剩余步数用于执行模型预测的动作。
            while t < max_steps + args.num_steps_wait:
                try:

                    # =================================================
                    # 7. 等待仿真物体稳定
                    # =================================================

                    # 在最初的若干步中，不调用 π0.5。
                    # 原因：仿真重置后，物体可能仍在下落或发生碰撞。
                    if t < args.num_steps_wait:

                        # LIBERO_DUMMY_ACTION：
                        # 预定义的占位动作，用于等待环境稳定，
                        # 不是模型预测的动作。
                        #
                        # env.step() 返回：
                        #   obs：执行动作后的新观测。
                        #   reward：当前步的奖励。
                        #   done：当前 Episode 是否结束。
                        #   info：额外信息。
                        obs, reward, done, info = env.step(
                            LIBERO_DUMMY_ACTION
                        )

                        # 仿真步数加一。
                        t += 1

                        # 跳过本次循环剩余部分，进入下一次循环。
                        continue

                    # =================================================
                    # 8. 获取并预处理两路 RGB 图像
                    # =================================================

                    # obs["agentview_image"]：
                    # 外部摄像头拍摄的场景图像。
                    #
                    # [::-1, ::-1]：
                    # 沿高度轴和宽度轴分别反转，相当于旋转 180°。
                    #
                    # np.ascontiguousarray()：
                    # 确保图像在内存中连续存储。
                    #
                    # 原始图像 Shape 通常为 [height, width, 3]。
                    img = np.ascontiguousarray(
                        obs["agentview_image"][::-1, ::-1]
                    )

                    # obs["robot0_eye_in_hand_image"]：
                    # 机器人末端附近的腕部摄像头图像。
                    # 同样旋转 180° 并转换为连续数组。
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )

                    # resize_with_pad()：
                    # 将图像缩放并填充至指定尺寸，同时保持宽高比。
                    #
                    # args.resize_size：目标图像的高度和宽度。
                    #
                    # convert_to_uint8()：
                    # 将图像转换为 uint8 类型。
                    #
                    # 例如 args.resize_size = 224：
                    #   img.shape       = [224, 224, 3]
                    #   wrist_img.shape = [224, 224, 3]
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            img,
                            args.resize_size,
                            args.resize_size,
                        )
                    )

                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            wrist_img,
                            args.resize_size,
                            args.resize_size,
                        )
                    )

                    # 保存当前外部摄像头图像，用于之后生成回放视频。
                    replay_images.append(img)

                    # =================================================
                    # 9. 判断是否需要请求新的动作序列
                    # =================================================

                    # action_plan 为空，说明之前预测的动作已经执行完毕。
                    # 此时需要根据最新观测重新请求模型推理。
                    if not action_plan:

                        # ---------------------------------------------
                        # 9.1 构造模型输入字典 element
                        # ---------------------------------------------

                        element = {
                            # 外部摄像头图像。
                            # Shape：[resize_size, resize_size, 3]
                            "observation/image": img,

                            # 腕部摄像头图像。
                            # Shape：[resize_size, resize_size, 3]
                            "observation/wrist_image": wrist_img,

                            # 机器人状态向量，由三部分拼接：
                            #
                            # 1. robot0_eef_pos：
                            #    末端执行器位置，Shape：[3]。
                            #
                            # 2. _quat2axisangle(robot0_eef_quat)：
                            #    将末端姿态从四元数转换为轴角表示，
                            #    Shape：[3]。
                            #
                            # 3. robot0_gripper_qpos：
                            #    夹爪位置状态，通常为 [2]。
                            #
                            # np.concatenate() 沿已有维度拼接：
                            # [3] + [3] + [2] -> [8]。
                            #
                            # 因此该 LIBERO 状态向量通常为 [8]。
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(
                                        obs["robot0_eef_quat"]
                                    ),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),

                            # 当前任务的自然语言描述。
                            # 例如：将指定物体放入某个容器。
                            "prompt": str(task_description),
                        }

                        # ---------------------------------------------
                        # 9.2 通过 WebSocket 请求模型推理
                        # ---------------------------------------------

                        # client.infer(element)：
                        # 将图像、机器人状态和语言指令发送给策略服务器。
                        #
                        # 服务器调用策略推理，并返回结果字典。
                        #
                        # ["actions"]：提取预测动作序列。
                        #
                        # action_chunk 的 Shape 通常为：
                        # [action_horizon, action_dim]。
                        #
                        # action_horizon：预测动作的时间步数。
                        # action_dim：每一步动作的维度。
                        action_chunk = client.infer(
                            element
                        )["actions"]

                        # 检查预测动作数量是否足够。
                        #
                        # args.replan_steps：
                        # 每次请求新动作序列后，计划连续执行的步数。
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), (
                            f"We want to replan every "
                            f"{args.replan_steps} steps, "
                            f"but policy only predicts "
                            f"{len(action_chunk)} steps."
                        )

                        # 将预测序列的前 args.replan_steps 步放入动作队列。
                        #
                        # 例如：
                        #   action_chunk.shape = [50, 7]
                        #   args.replan_steps = 5
                        #
                        #   action_chunk[:5].shape = [5, 7]
                        #
                        # 注意：这里只取前 5 步，不会执行整个 50 步序列。
                        action_plan.extend(
                            action_chunk[:args.replan_steps]
                        )

                    # =================================================
                    # 10. 从动作队列中取出下一步动作
                    # =================================================

                    # popleft()：
                    # 取出队首动作，并将其从队列中删除。
                    #
                    # action.shape = [action_dim]
                    action = action_plan.popleft()

                    # =================================================
                    # 11. 将动作发送给仿真环境执行
                    # =================================================

                    # action.tolist()：
                    # 将 NumPy 数组转换为 Python 列表。
                    #
                    # env.step()：
                    # 将动作传给 LIBERO 仿真环境中的机器人控制器，
                    # 推进仿真并获取执行后的观测。
                    #
                    # 返回：
                    #   obs：新的环境观测。
                    #   reward：当前步奖励。
                    #   done：任务是否结束。
                    #   info：额外信息。
                    obs, reward, done, info = env.step(
                        action.tolist()
                    )

                    # 如果 done 为 True，则当前任务 Episode 已结束。
                    # 此处将其记为成功，并退出当前 Episode。
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    # 当前仿真步数加一。
                    t += 1

                except Exception as e:

                    # 捕获当前 Episode 执行过程中发生的异常。
                    # 记录错误日志，并结束当前 Episode 的循环。
                    logging.error(f"Caught exception: {e}")
                    break

            # ========================================================
            # 12. 更新 Episode 统计量
            # ========================================================

            # 当前任务完成的 Episode 数量加一。
            task_episodes += 1

            # 所有任务累计完成的 Episode 数量加一。
            total_episodes += 1

            # ========================================================
            # 13. 保存当前 Episode 的回放视频
            # ========================================================

            # 根据 done 生成视频文件名中的结果标记。
            suffix = "success" if done else "failure"

            # 将任务描述中的空格替换为下划线，方便生成文件名。
            task_segment = task_description.replace(" ", "_")

            # 将 replay_images 中的图像写入 MP4 视频。
            #
            # 输出文件名示例：
            # rollout_pick_up_the_object_success.mp4
            #
            # fps=10：视频帧率为每秒 10 帧。
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # ========================================================
            # 14. 输出当前 Episode 的评测结果
            # ========================================================

            # 当前 Episode 是否成功。
            logging.info(f"Success: {done}")

            # 所有任务累计完成的 Episode 数量。
            logging.info(
                f"# episodes completed so far: {total_episodes}"
            )

            # 累计成功次数与总体成功率。
            logging.info(
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        # ============================================================
        # 15. 输出当前任务和整体任务集的成功率
        # ============================================================

        # 当前任务成功率：
        # 当前任务成功的 Episode 数 / 当前任务总 Episode 数。
        logging.info(
            f"Current task success rate: "
            f"{float(task_successes) / float(task_episodes)}"
        )

        # 所有任务累计成功率：
        # 所有任务成功的 Episode 数 / 所有任务总 Episode 数。
        logging.info(
            f"Current total success rate: "
            f"{float(total_successes) / float(total_episodes)}"
        )

    # ================================================================
    # 16. 输出整个任务集的最终评测结果
    # ================================================================

    # 所有任务、所有 Episode 的最终成功率。
    logging.info(
        f"Total success rate: "
        f"{float(total_successes) / float(total_episodes)}"
    )

    # 所有任务累计完成的 Episode 数量。
    logging.info(f"Total episodes: {total_episodes}")

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
