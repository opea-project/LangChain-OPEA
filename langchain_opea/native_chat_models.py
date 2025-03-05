"""Native Chat Wrapper."""

from typing import Any, AsyncIterator, Iterator, List, Optional
import logging
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from pydantic import model_validator
from typing_extensions import Self
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""
DEFAULT_MODEL_ID = "Intel/neural-chat-7b-v3-3"
logger = logging.getLogger(__name__)

class AttributeContainer:
    def __init__(self, **kwargs):
        # Set attributes dynamically based on keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)


args = AttributeContainer(
    device="hpu",
    model_name_or_path=DEFAULT_MODEL_ID,
    bf16=True,
    max_new_tokens=100,
    max_input_tokens=0,
    batch_size=1,
    warmup=3,
    n_iterations=5,
    local_rank=0,
    use_kv_cache=True,
    use_hpu_graphs=True,
    dataset_name=None,
    column_name=None,
    do_sample=False,
    num_beams=1,
    trim_logits=False,
    seed=27,
    profiling_warmup_steps=0,
    profiling_steps=0,
    profiling_record_shapes=False,
    prompt=None,
    bad_words=None,
    force_words=None,
    assistant_model=None,
    peft_model=None,
    token=None,
    model_revision="main",
    attn_softmax_bf16=False,
    output_dir=None,
    bucket_size=-1,
    dataset_max_samples=-1,
    limit_hpu_graphs=False,
    reuse_cache=False,
    verbose_workers=False,
    simulate_dyn_prompt=None,
    reduce_recompile=False,
    use_flash_attention=False,
    flash_attention_recompute=False,
    flash_attention_causal_mask=False,
    flash_attention_fast_softmax=False,
    book_source=False,
    torch_compile=False,
    ignore_eos=True,
    temperature=1.0,
    top_p=1.0,
    const_serialization_path=None,
    csp=None,
    disk_offload=False,
    trust_remote_code=False,
    quant_config=os.getenv("QUANT_CONFIG", ""),
    num_return_sequences=1,
    bucket_internal=False,
)

class ChatNative(BaseChatModel):
    """
    Wrapper for using LLMs run on Intel Gaudi as ChatModels.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatNative

            chat_model = ChatNative(model_name="Intel/neural-chat-7b-v3-3")

    Adapted from: https://python.langchain.com/docs/integrations/chat/llama2_chat
    """

    llm: Any
    """LLM, must be of type HuggingFacePipeline
        """
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_id: Optional[str] = None
    device: Optional[str] = "hpu"
    model_name: str = Field(alias="model", default=DEFAULT_MODEL_ID)

    streaming: bool = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        args.model_name_or_path = self.model_name
        if self.device == "hpu":
            pipe = GaudiTextGenerationPipeline(
                args,
                logger,
                use_with_langchain=True
            )
            hfpipe = HuggingFacePipeline(pipeline=pipe)
            self.llm = hfpipe
            self.tokenizer = pipe.tokenizer
        else:
            raise NotImplementedError(f"Only support hpu device now, device {self.device} not supported.")


    @model_validator(mode="after")
    def validate_llm(self) -> Self:
        if not isinstance(
            self.llm,
            (HuggingFacePipeline),
        ):
            raise TypeError(
                "Expected llm to be one of HuggingFacePipeline"
                f", received {type(self.llm)}"
            )
        return self

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._to_chat_prompt(messages)

        for data in self.llm.stream(request, **kwargs):
            delta = data
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        request = self._to_chat_prompt(messages)
        async for data in self.llm.astream(request, **kwargs):
            delta = data
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                await run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    @property
    def _llm_type(self) -> str:
        return "gaudi-chat-wrapper"

