"""Native Chat Wrapper."""

from typing import Any, AsyncIterator, Iterator, List, Optional

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

class ChatNative(BaseChatModel):
    """
    Wrapper for using LLMs run on Intel Gaudi as ChatModels.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatNative
            from integrations.pipeline import GaudiTextGenerationPipeline
            from integrations.gaudiutils import initialize_model

            args.model_name_or_path = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
            args.max_new_tokens = 512 
            args.use_hpu_graphs = True
            args.use_kv_cache = True

            pipe = GaudiTextGenerationPipeline(
                args,
                logger, 
                use_with_langchain=True
            )
            hfpipe = HuggingFacePipeline(pipeline=pipe)
            model, _, tokenizer, _= initialize_model(args, logger)

            chat_model = ChatNative(
                temperature=0.1, 
                llm=hfpipe, 
                tokenizer=tokenizer
            )

    Adapted from: https://python.langchain.com/docs/integrations/chat/llama2_chat
    """

    llm: Any
    """LLM, must be of type HuggingFacePipeline
        """
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_id: Optional[str] = None
    streaming: bool = False

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer

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

