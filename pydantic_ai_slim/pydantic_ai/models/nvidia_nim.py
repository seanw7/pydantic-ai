from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast, overload

from pydantic import ValidationError
from typing_extensions import assert_never

from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from pydantic_ai._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from pydantic_ai._run_context import RunContext
from pydantic_ai._thinking_part import split_content_into_text_and_thinking
from pydantic_ai._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc, number_to_datetime
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
    download_item,
    get_user_agent,
)
from pydantic_ai.profiles import ModelProfile, ModelProfileSpec
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

try:
    from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, AsyncStream
    from openai.types import chat
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartInputAudioParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
    )
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL
    from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
    from openai.types.chat.chat_completion_content_part_param import File, FileFile
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the Nvidia NIM model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'NvidiaNIMModel',
    'NvidiaNIMModelSettings',
    'NvidiaNIMModelName',
)

NvidiaNIMModelName = str
"""A string representing the Nvidia NIM model name to be used."""


class NvidiaNIMModelSettings(ModelSettings, total=False):
    """Settings used for an Nvidia NIM model request."""

    nvidia_min_p: float
    nvidia_prompt_logprobs: int
    nvidia_add_special_tokens: bool
    nvidia_documents: list[dict[str, str]]
    nvidia_chat_template: str
    nvidia_chat_template_kwargs: dict[str, Any]
    nvidia_guided_whitespace_pattern: str
    nvidia_ignore_eos: bool
    nvidia_repetition_penalty: float
    nvidia_top_k: int
    nvidia_guided_choice: list[str]
    nvidia_guided_json: str | object
    nvidia_guided_regex: str
    nvidia_guided_grammar: str
    nvidia_guided_decoding_backend: str


@dataclass(init=False)
class NvidiaNIMModel(Model):
    """A client for interacting with NVIDIA NIM models."""

    client: AsyncOpenAI = field(repr=False)
    _model_name: NvidiaNIMModelName = field(repr=False)
    _system: str = field(default='nvidia', repr=False)

    def __init__(
        self,
        model_name: NvidiaNIMModelName,
        *,
        provider: Literal['nvidia'] | Provider[AsyncOpenAI] = 'nvidia',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, False, cast(NvidiaNIMModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        model_response.usage.requests = 1
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._completions_create(
            messages,
            True,
            cast(NvidiaNIMModelSettings, model_settings or {}),
            model_request_parameters,
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @property
    def model_name(self) -> NvidiaNIMModelName:
        return self._model_name

    @property
    def system(self) -> str:
        return self._system

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: NvidiaNIMModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: NvidiaNIMModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: NvidiaNIMModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)
        tool_choice: Literal['none', 'required', 'auto'] | None = 'auto' if tools else None

        openai_messages = await self._map_messages(messages)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output:
            response_format = {'type': 'json_object'}

        extra_body: dict[str, Any] = {}
        nvext: dict[str, Any] = {}

        top_level_params = {
            'nvidia_min_p': 'min_p',
            'nvidia_prompt_logprobs': 'prompt_logprobs',
            'nvidia_add_special_tokens': 'add_special_tokens',
            'nvidia_documents': 'documents',
            'nvidia_chat_template': 'chat_template',
            'nvidia_chat_template_kwargs': 'chat_template_kwargs',
            'nvidia_guided_whitespace_pattern': 'guided_whitespace_pattern',
        }
        nvext_params = {
            'nvidia_ignore_eos': 'ignore_eos',
            'nvidia_repetition_penalty': 'repetition_penalty',
            'nvidia_top_k': 'top_k',
            'nvidia_guided_choice': 'guided_choice',
            'nvidia_guided_json': 'guided_json',
            'nvidia_guided_regex': 'guided_regex',
            'nvidia_guided_grammar': 'guided_grammar',
            'nvidia_guided_decoding_backend': 'guided_decoding_backend',
        }

        for setting_key, api_key in top_level_params.items():
            if (value := model_settings.get(setting_key)) is not None:
                extra_body[api_key] = value

        for setting_key, api_key in nvext_params.items():
            if (value := model_settings.get(setting_key)) is not None:
                nvext[api_key] = value

        if nvext:
            extra_body['nvext'] = nvext

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stream=stream,
                stream_options={'include_usage': True} if stream else NOT_GIVEN,
                stop=model_settings.get('stop_sequences', NOT_GIVEN),
                max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                response_format=response_format or NOT_GIVEN,
                seed=model_settings.get('seed', NOT_GIVEN),
                user=model_settings.get('user', NOT_GIVEN),
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
                frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
                logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
                logprobs=model_settings.get('logprobs', NOT_GIVEN),
                top_logprobs=model_settings.get('top_logprobs', NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=extra_body or model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        if not isinstance(response, chat.ChatCompletion):
            raise UnexpectedModelBehavior('Invalid response from NIM chat completions endpoint, expected JSON data')

        timestamp = number_to_datetime(response.created) if response.created else _now_utc()

        try:
            response = chat.ChatCompletion.model_validate(response.model_dump())
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from NIM chat completions endpoint: {e}') from e

        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        if choice.message.content is not None:
            items.extend(split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id))

        return ModelResponse(
            items,
            usage=_map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            vendor_id=response.id,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk], model_request_parameters: ModelRequestParameters
    ) -> NvidiaNIMStreamedResponse:
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        return NvidiaNIMStreamedResponse(
            _model_name=self._model_name,
            _model_profile=self.profile,
            _response=peekable_response,
            _timestamp=number_to_datetime(first_chunk.created),
            model_request_parameters=model_request_parameters,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.output_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.output_tools]
        return tools

    def _map_tool_definition(self, f: ToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }

    async def _map_messages(self, messages: list[ModelMessage]) -> list[chat.ChatCompletionMessageParam]:
        openai_messages: list[chat.ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    openai_messages.append(item)
            elif isinstance(message, ModelResponse):
                # FIX: Add explicit types for the lists
                texts: list[str] = []
                tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ThinkingPart):
                        pass
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
                if texts:
                    message_param['content'] = '\n\n'.join(texts)
                if tool_calls:
                    message_param['tool_calls'] = tool_calls
                openai_messages.append(message_param)
        if instructions := self._get_instructions(messages):
            openai_messages.insert(0, chat.ChatCompletionSystemMessageParam(content=instructions, role='system'))

        return openai_messages

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool', tool_call_id=_guard_tool_call_id(t=part), content=part.model_response_str()
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool', tool_call_id=_guard_tool_call_id(t=part), content=part.model_response()
                    )
            else:
                assert_never(part)

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url = ImageURL(url=item.url)
                    content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    if item.is_image:
                        image_url = ImageURL(url=f'data:{item.media_type};base64,{base64_encoded}')
                        content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    elif item.is_audio:
                        assert item.format in ('wav', 'mp3')
                        audio = InputAudio(data=base64_encoded, format=item.format)
                        content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                    elif item.is_document:
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=f'data:{item.media_type};base64,{base64_encoded}',
                                    filename=f'filename.{item.format}',
                                ),
                                type='file',
                            )
                        )
                    else:
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    downloaded_item = await download_item(item, data_format='base64', type_format='extension')
                    assert downloaded_item['data_type'] in ('wav', 'mp3')
                    audio = InputAudio(data=downloaded_item['data'], format=downloaded_item['data_type'])
                    content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                elif isinstance(item, DocumentUrl):
                    downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                    file = File(
                        file=FileFile(
                            file_data=downloaded_item['data'], filename=f'filename.{downloaded_item["data_type"]}'
                        ),
                        type='file',
                    )
                    content.append(file)
                elif isinstance(item, VideoUrl):
                    raise NotImplementedError('VideoUrl is not supported for NVIDIA NIM')
                else:
                    assert_never(item)
        return chat.ChatCompletionUserMessageParam(role='user', content=content)

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
        return chat.ChatCompletionMessageToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    def _map_json_schema(self, o: OutputObjectDefinition) -> chat.completion_create_params.ResponseFormat:
        json_schema_part: dict[str, Any] = {
            'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
            'schema': o.json_schema,
        }
        if o.description:
            json_schema_part['description'] = o.description

        strict = True
        if (
            OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition
            and o.strict is not None
        ):
            strict = o.strict
        json_schema_part['strict'] = strict

        # FIX: Use cast to assert the final structure conforms to the TypedDict.
        return cast(
            chat.completion_create_params.ResponseFormat,
            {
                'type': 'json_schema',
                'json_schema': json_schema_part,
            },
        )

    @staticmethod
    def _get_instructions(messages: list[ModelMessage]) -> str | None:
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        return part.content
        return None


@dataclass
class NvidiaNIMStreamedResponse(StreamedResponse):
    _model_name: NvidiaNIMModelName
    _model_profile: ModelProfile
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += _map_usage(chunk)
            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            if content := choice.delta.content:
                thinking_tags = getattr(self._model_profile, 'thinking_tags', ('<thinking>', '</thinking>'))
                # FIX: Use keyword arguments to match the working implementation.
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content', content=content, thinking_tags=thinking_tags
                )
                if maybe_event:
                    yield maybe_event

            for dtc in choice.delta.tool_calls or []:
                # FIX: Use keyword arguments here as well.
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event:
                    yield maybe_event

    @property
    def model_name(self) -> NvidiaNIMModelName:
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        return self._timestamp


def _map_usage(response: chat.ChatCompletion | ChatCompletionChunk) -> usage.Usage:
    response_usage = response.usage
    if response_usage is None:
        return usage.Usage()
    return usage.Usage(
        requests=1,
        request_tokens=response_usage.prompt_tokens,
        response_tokens=response_usage.completion_tokens,
        total_tokens=response_usage.total_tokens,
    )
