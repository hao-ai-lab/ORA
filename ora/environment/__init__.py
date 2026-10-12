"""Environment extension contracts and registered session construction."""

from ora.contracts.action import (
    ActionValidationError,
    MOTION_ACTION_SCHEMA,
    MOTION_ACTION_V3_SCHEMA,
    RUNTIME_ACTION_SCHEMA,
    TARGET_STATE_ACTION_SCHEMA,
    target_state_phases,
    validate_runtime_action,
)
from .base import (
    ENVIRONMENT_SESSION_SCHEMA,
    ENVIRONMENT_STATUS_SCHEMA,
    EnvironmentContractError,
    RestartableEnvironmentSession,
    validate_restartable_environment_session,
    validate_environment_session,
    validate_environment_status,
)
from .contracts import EnvironmentSession
from .definitions import (
    ENVIRONMENT_DESCRIPTOR_SCHEMA,
    ENVIRONMENT_DESCRIPTOR_V2_SCHEMA,
    EnvironmentDefinition,
    EnvironmentDefinitionError,
    EnvironmentRuntimeDescriptor,
    environment_definition,
    environment_definitions,
    resolve_runtime_descriptor,
    validate_runtime_extension,
)
from .factory import EnvironmentFactoryError, build_environment_session
from .rpc import EnvironmentRpcError, LocalRpcEnvironmentSession, RPC_SCHEMA
from .session import DescribedEnvironmentSession
from .synthetic import SyntheticSession, synthetic_public_environment_context

__all__ = [
    "ActionValidationError", "ENVIRONMENT_DESCRIPTOR_SCHEMA",
    "ENVIRONMENT_DESCRIPTOR_V2_SCHEMA",
    "ENVIRONMENT_SESSION_SCHEMA", "ENVIRONMENT_STATUS_SCHEMA",
    "DescribedEnvironmentSession", "EnvironmentContractError",
    "EnvironmentDefinition", "EnvironmentDefinitionError",
    "EnvironmentFactoryError", "EnvironmentRpcError",
    "EnvironmentRuntimeDescriptor", "EnvironmentSession",
    "RestartableEnvironmentSession",
    "LocalRpcEnvironmentSession", "RPC_SCHEMA", "RUNTIME_ACTION_SCHEMA",
    "MOTION_ACTION_SCHEMA", "MOTION_ACTION_V3_SCHEMA",
    "SyntheticSession", "TARGET_STATE_ACTION_SCHEMA",
    "build_environment_session", "environment_definition",
    "environment_definitions", "resolve_runtime_descriptor",
    "synthetic_public_environment_context", "target_state_phases",
    "validate_environment_session", "validate_environment_status",
    "validate_restartable_environment_session", "validate_runtime_action",
    "validate_runtime_extension",
]
