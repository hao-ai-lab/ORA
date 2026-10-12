"""Deterministic strict-provider names with canonical transcript identities."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError

_STRICT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def tool_alias(name: str) -> str:
    if _STRICT_NAME.fullmatch(name):
        return name
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if not slug or not slug[0].isalpha():
        slug = "tool_" + slug
    return slug[:55] + "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]


def openai_tool_names(request: Mapping[str, Any]) -> dict[str, str]:
    """Include retained tool turns even when a later phase changes its tools."""
    names = []
    for tool in request.get("tools", []):
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping):
            names.append(tool["function"].get("name"))
    for message in request.get("messages", []):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "tool":
            names.append(message.get("name"))
        for call in message.get("tool_calls", []):
            if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping):
                names.append(call["function"].get("name"))
    choice = request.get("tool_choice")
    if isinstance(choice, Mapping) and isinstance(choice.get("function"), Mapping):
        names.append(choice["function"].get("name"))
    aliases = {}
    for name in names:
        if not isinstance(name, str) or not name:
            continue
        alias = tool_alias(name)
        if aliases.setdefault(alias, name) != name:
            raise PolicySessionError("OpenAI tool-name alias collision")
    return aliases


def project_openai_tool_names(body: dict[str, Any], aliases: Mapping[str, str]) -> None:
    """Project a copied, continuation-validated wire body in place."""
    for tool in body.get("tools", []):
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str) and function["name"]:
            function["name"] = tool_alias(function["name"])
    # A nameless rejected call stays nameless: inventing an alias for an empty
    # name would also rewrite the earlier alias-disclosure prefix on continuation.
    for message in body["messages"]:
        if message.get("role") == "tool" and isinstance(message.get("name"), str):
            message["name"] = tool_alias(message["name"])
        for call in message.get("tool_calls", []):
            function = call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str) and function["name"]:
                function["name"] = tool_alias(function["name"])
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        name = choice["function"].get("name")
        if isinstance(name, str):
            choice["function"]["name"] = tool_alias(name)
    renamed = {name: alias for alias, name in aliases.items() if alias != name}
    if renamed:
        # Keep policy instructions and tool results intact; disclose the mapping
        # in the provider prefix just as the Anthropic projection does.
        index = next((i for i, message in enumerate(body["messages"])
            if message.get("role") != "system"), len(body["messages"]))
        body["messages"].insert(index, {"role":"system", "content":
            "<wire_tool_aliases>\n" + json.dumps(renamed, ensure_ascii=False, sort_keys=True)
            + "\n</wire_tool_aliases>"})
