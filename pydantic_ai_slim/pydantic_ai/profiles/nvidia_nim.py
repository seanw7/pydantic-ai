from __future__ import annotations as _annotations

import re
from dataclasses import dataclass

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


@dataclass
class NvidiaNIMModelProfile(ModelProfile):
    """Profile for models used with NvidiaNIMModel.

    Since the Nvidia NIM API is designed for OpenAI compatibility, many of these
    settings mirror the OpenAI profile. They are prefixed with `nvidia_` to
    allow for safely merging this profile with others.
    """

    nvidia_supports_strict_tool_definition: bool = True
    """
    Indicates that the NIM API supports well-defined, strict tool definitions,
    which is crucial for reliable function calling.
    """

    nvidia_supports_sampling_settings: bool = True
    """
    NIM models support sampling parameters like `temperature` and `top_p`,
    so this is enabled by default.
    """

    nvidia_supports_tool_choice_required: bool = True
    """
    Whether the provider accepts the value ``tool_choice='required'`` in the
    request payload. NIM supports this.
    """


def nvidia_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for an Nvidia NIM model.

    This function returns a default profile for NIM models, enabling key features
    like JSON schema output and providing a schema transformer to ensure
    compatibility.

    Args:
        model_name: The name of the Nvidia NIM model (currently unused, but
                    kept for API consistency and future customization).

    Returns:
        A configured NvidiaNIMModelProfile instance.
    """
    return NvidiaNIMModelProfile(
        json_schema_transformer=NvidiaNIMJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


# These constants and the transformer class are adapted from the OpenAI profile.
# They enforce a schema structure that is robust and compatible with most
# OpenAI-like APIs, which is ideal for Nvidia NIM.

_STRICT_INCOMPATIBLE_KEYS = [
    'minLength',
    'maxLength',
    'patternProperties',
    'unevaluatedProperties',
    'propertyNames',
    'minProperties',
    'maxProperties',
    'unevaluatedItems',
    'contains',
    'minContains',
    'maxContains',
    'uniqueItems',
]

_STRICT_COMPATIBLE_STRING_FORMATS = [
    'date-time',
    'time',
    'date',
    'duration',
    'email',
    'hostname',
    'ipv4',
    'ipv6',
    'uuid',
]

_sentinel = object()


@dataclass
class NvidiaNIMJsonSchemaTransformer(JsonSchemaTransformer):
    """Recursively transforms a JSON schema to make it compatible with the strict validation expected by Nvidia NIM and other OpenAI-compatible APIs.

    This ensures that tool definitions are unambiguous for the model by:
    - Setting `additionalProperties` to `false` for all objects.
    - Marking all properties of an object as `required`.
    - Removing or adapting schema keys that are not supported in strict mode.
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict)
        self.root_ref = schema.get('$ref')

    def walk(self) -> JsonSchema:
        result = super().walk()
        if self.root_ref is not None:
            result.pop('$ref', None)
            root_key = re.sub(r'^#/\$defs/', '', self.root_ref)
            result.update(self.defs.get(root_key) or {})
        return result

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema = self._remove_metadata_keys(schema)
        schema = self._ensure_strict_mode(schema)
        schema = self._fix_unsupported_keys(schema)
        return schema

    def _remove_metadata_keys(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('description', None)
        return schema

    def _ensure_strict_mode(self, schema: JsonSchema) -> JsonSchema:
        if 'properties' in schema:
            schema['additionalProperties'] = False
            # Mark all props as required
            required = schema.get('required', [])
            required = list(set(required + list(schema['properties'].keys())))
            schema['required'] = required
        return schema

    def _fix_unsupported_keys(self, schema: JsonSchema) -> JsonSchema:
        # Handle unsupported keys like $schema, $id, etc.
        for key in ['$schema', '$id', '$anchor']:
            schema.pop(key, None)
        return schema
