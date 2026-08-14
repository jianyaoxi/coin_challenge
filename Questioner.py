from abc import ABC, abstractmethod
import numpy as np
from utils import ClientBasedLLM
from retrying import retry
import time
import re
from retrying import retry

# Example of a questioner prompt
QUESTIONER_EXAMPLE_PROMPT = (
    "An oracle has a fixed target image and has given you this description of the object in the target image: {TARGET_DESCRIPTION}. "
    "You are given the above image, which may or may not picturing the same object as the target image owned by the oracle. Your goal is to decide whether the "
    "image that you see corresponds to the same image target image owned by the oracle. "
    "Use, to guide your decision, the description of the object, the give image, and, if they exists, the questions that you previously asked to the oracle "
    "and the associated answers. Provide a reasoning about your conclusion, or why you are uncertain and asked a question. "
    "If you are sure that the two image match, return the score 2, if you are sure that they don't match, return "
    "the score 0. If you are unsure either way, return the score 1 and ask an informative question to the oracle about what it might appears in the target image, "
    "to dispel your doubts. You can always trust the oracle's answers and the initial description. Do not ask questions that can be directly answered by "
    "reading the initial description, or questions about the image that you are provided. Be careful: the target image and the given image might differ only in some small details, "
    "like the color of the object, its texture, or the presence of other objects. For example, the target image might picture a bed with a blue comforter, "
    "and the given image a bed with a red comforter, or a blue bed but with a white comforter. "
    "The image might have distortions or digital artifacts: *NEVER* mention them in the question. Prefer asking question if the description is very generic. "
    "Strictly follow this output format: "
    "<motivation>Your reasoning here (under 60 words, do NOT use double quotes \")</motivation><score>0, 1, or 2</score><question>Your question or '' (if score is not 1)</question>"
)


def _validate_observation(observation):
    assert isinstance(observation["image"], np.ndarray) and (
        not observation["answer"] or isinstance(observation["answer"], str)
    ), (
        "Wrong observation format: it must be a dictionary with keys 'image' and 'answer', where 'answer' is a numpy array and 'answer' is either a string or None"
    )
    assert (
        len(observation["image"].shape) == 3 and observation["image"].shape[2] == 3
    ), "Wrong image format: must be a numpy array of shape (H,W,3) --- an rgb image."


class QuestionerInterface(ABC):
    """Abstract Questioner class. Your questioner should inherit from this."""

    def __init__(self, info, *args):
        self.info = info  # required info like the task description
        self.target_description = info["task_description"]

    @abstractmethod
    def ask_or_conclude(self, observation):
        # TODO: this is what you have to implement
        pass

    def add_answer(self, answer):
        self.answers.append(answer)

    def reset_questions(self):
        self.questions = []
        self.answers = []

    def reset_time(self):
        self.time_required = 0


class QuestionerLocalVLM(QuestionerInterface):
    """Simple class that can use a local VLM (run via VLLM) as the questioner."""

    def __init__(self, info, model_id: str):
        # info will contain the target object description: info["target_description"]
        # This is also saved in self.target_description
        super().__init__(info)
        self.client = ClientBasedLLM(model_id=model_id)  # Handle the VLLM connection
        self.questions = []
        self.reasonings = []
        self.answers = []
        self.time_required = 0
        self.n_questions = 0

    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=2000,
        wait_exponential_max=60000,
    )
    def ask_or_conclude(self, observation):
        _validate_observation(observation)
        start_time = time.time()

        # Define `prompt_to_use`` and other stuff here. The prompt can ask the model to evaluate the observation vs the description,
        # in order to reason about it and conclude (or not) whether the object in the image matches the description.
        # You can also handle resource-keeping tasks, like keeping track of previous questions and answers (although you don't need to)
        prompt_to_use = "TODO"  # TODO:

        # prompt_to_use = QUESTIONER_EXAMPLE_PROMPT.format(TARGET_DESCRIPTION=self.target_description)

        response = self.client.ask(
            prompt=prompt_to_use,
            images=[observation["image"]],
        )

        end_time = time.time()

        # Parse and return a question or a conclusion
        # TODO: parse the question/conclusion and return it.
        ## Return either (if uncertain whether the observation corresponds to the target description or not)
        # return dict(question=question, conclusion=None, reasoning=reasoning)
        ## if certain that is a match
        # return dict(question=None, conclusion=1, reasoning=reasoning)
        ## or if certaint that is NOT a match
        # return dict(question=None, conclusion=0, reasoning=reasoning)
        raise NotImplementedError("Impement this function.")


class YourQuestioner(QuestionerInterface):
    def __init__(self, info, *args):
        super().__init__(info)
        # 从可变参数获取模型ID，若未提供则使用默认值（可自行替换）
        if args:
            model_id = args[0]
        else:
            # 默认使用微调后的 LoRA 模型（vLLM 里 --lora-modules qwen32b=<adapter>）
            # 未微调底座为 "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
            model_id = "qwen32b"
        self.client = ClientBasedLLM(model_id=model_id)
        
        # ---------- 记录所有历史 ----------
        self.questions = []      # 存储所有已问的问题
        self.answers = []        # 存储所有已收到的答案（由外部 add_answer 添加）
        self.reasonings = []     # 存储每次的推理过程（可选）
        self.time_required = 0   # 累计推理时间
        self.n_questions = 0     # 累计提问次数

    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=2000,
        wait_exponential_max=60000,
    )
    def ask_or_conclude(self, observation):
        _validate_observation(observation)
        start_time = time.time()

        # ----- 1. 构造基础提示（包含目标描述） -----
        base_prompt = QUESTIONER_EXAMPLE_PROMPT.format(
            TARGET_DESCRIPTION=self.target_description
        )

        # ----- 2. 构建历史对话上下文（问题和答案“拉链”在一起） -----
        history = ""
        if self.questions and self.answers:
            # 确保两者长度一致，若不等则截断至较短者（正常情况下应一致）
            history += "\nPreviously, you have asked the oracle the following questions and received these answers:\n"
            for idx, (q, a) in enumerate(zip(self.questions, self.answers)):
                history += f"  Q{idx+1}: {q}\n"
                history += f"  A{idx+1}: {a}\n"
            history += "\nNow, using this previous information, please decide on the current image.\n"

        # 最终提示 = 基础提示 + 历史（如果有）
        prompt_to_use = base_prompt + history

        # ----- 3. 调用本地 VLM（通过 vLLM） -----
        response = self.client.ask(
            prompt=prompt_to_use,
            images=[observation["image"]],
        )

        end_time = time.time()
        self.time_required += (end_time - start_time)

        # ----- 4. 解析模型响应 -----
        motivation_match = re.search(r'<motivation>(.*?)</motivation>', response, re.DOTALL)
        score_match = re.search(r'<score>([0-2])</score>', response)
        question_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)

        reasoning = motivation_match.group(1).strip() if motivation_match else ""
        score = int(score_match.group(1)) if score_match else -1
        question = question_match.group(1).strip() if question_match else ""

        self.reasonings.append(reasoning)

        # ----- 5. 根据分数决定返回结论还是继续提问 -----
        if score == 2:          # 确信匹配
            return dict(question=None, conclusion=1, reasoning=reasoning)
        elif score == 0:        # 确信不匹配
            return dict(question=None, conclusion=0, reasoning=reasoning)
        else:                   # 不确定（score=1 或解析失败），需要提问
            if question:
                # ---------- 将新问题记录到 self.questions 中 ----------
                self.questions.append(question)
                self.n_questions += 1
                return dict(question=question, conclusion=None, reasoning=reasoning)
            else:
                # 若未提取到有效问题，则生成一个后备问题，同样记录
                fallback = "Could you provide more details about the target image?"
                self.questions.append(fallback)
                self.n_questions += 1
                return dict(question=fallback, conclusion=None, reasoning=reasoning)