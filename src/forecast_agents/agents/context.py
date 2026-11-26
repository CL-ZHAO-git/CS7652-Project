"""Build a bounded conversation view without splitting tool-call exchanges."""

import json

from forecast_agents.llm.types import Message, as_json

PROTOCOL_FILES = {"task.json", "pipeline_contract.md"}


def context_size(messages):
    # Conservative character measure, not a provider token count. Native content
    # may duplicate normalized fields; retaining it is necessary for continuation.
    return sum(len(json.dumps(as_json(m), ensure_ascii=False)) for m in messages)


def working_state(history):
    """Remember observed source previews and checks, including cache invalidation.

    This is deterministic extraction of existing observations, not a generated
    claim that a stage passed. Successes and failures remain distinguishable.
    """
    cache, executions, writes = {}, {}, []
    pending = {}
    revision = 0
    validation = None
    description = None
    audit = None
    for message in history:
        if message.role == "assistant":
            pending = {call.id: call for call in message.tool_calls}
            continue
        if message.role != "tool":
            continue
        call = pending.get(message.tool_call_id)
        if call is None:
            continue
        result = json.loads(message.text)
        if result.get("policy_blocked"):
            continue
        path = call.arguments.get("path", "")
        if call.name == "write_file" and result.get("written") == path:
            # Writing a report after a check must not make that check appear stale.
            # Track observed code edits and candidate-file edits conservatively.
            if path.casefold().endswith(".py") or path.casefold().startswith("solution/"):
                revision += 1
            writes.append(path)
            # All older pages of an edited file are stale, including nonzero offsets.
            cache = {key: value for key, value in cache.items() if key[0] != path}
            if path.casefold().endswith(".py") or path.casefold() in PROTOCOL_FILES:
                content = call.arguments["content"]
                cache[(path, 0)] = {"path": path, "offset": 0, "text": content[:8000],
                                    "truncated": len(content) > 8000, "source": "successful_write"}
        elif call.name == "read_file" and not result.get("is_error") and "text" in result:
            if (path.casefold().startswith("data/") or
                    not (path.casefold().endswith(".py") or path.casefold() in PROTOCOL_FILES)):
                continue
            offset = call.arguments.get("offset", 0)
            cache.pop((path, offset), None)
            cache[(path, offset)] = {"path": path, "offset": offset, "text": result["text"][:8000],
                                   "truncated": result.get("truncated", False) or len(result["text"]) > 8000,
                                   "source": "successful_read"}
        elif call.name == "run_python":
            executions.pop(path, None)
            executions[path] = {"path": path, "at_write_revision": revision,
                                "is_error": bool(result.get("is_error")),
                                **{k: result[k] for k in ("returncode", "timed_out", "error", "error_type", "locations", "output_withheld", "diagnostic_code", "operand_types", "missing_key", "hint") if k in result},
                                "stdout_preview": result.get("stdout", "")[:1000],
                                "stderr_preview": result.get("stderr", "")[:500],
                                "stdout_truncated": len(result.get("stdout", "")) > 1000 or result.get("stdout_truncated", False),
                                "stderr_truncated": len(result.get("stderr", "")) > 500 or result.get("stderr_truncated", False)}
        elif call.name == "validate_candidate":
            validation = {k: result[k] for k in ("report", "candidate_hash", "metrics", "is_error", "error",
                          "error_type", "locations", "diagnostic_code", "operand_types", "missing_key", "hint", "operation") if k in result}
            validation["at_write_revision"] = revision
        elif call.name == "describe_data" and not result.get("is_error"):
            description = result
        elif call.name == "audit_candidate":
            audit = result
    protocol = [v for (path, offset), v in cache.items() if path.casefold() in PROTOCOL_FILES]
    recent = [v for (path, offset), v in cache.items() if path.casefold() not in PROTOCOL_FILES][-2:]
    return {"successful_write_paths": list(dict.fromkeys(writes))[-12:],
            "current_write_revision": revision,
            "cached_file_previews": protocol + recent,
            "recent_executions": list(executions.values())[-3:],
            "data_description": description,
            "latest_audit_result": audit,
            "latest_validation_result": validation}


def request_context(history, notice, max_chars):
    """Keep the current task and recent complete exchanges; never mutate history.

    The latest exchange is always retained, even if it alone exceeds the target.
    Full history remains in the controller and original events remain on disk.
    """
    current = max(i for i, m in enumerate(history) if m.role == "user")
    prefix = [history[0], notice, history[current]]
    groups = []
    for message in history[current + 1:]:
        if message.role == "assistant":
            groups.append([message])
        elif message.role == "tool" and groups:
            groups[-1].append(message)
        else:
            raise ValueError("Conversation is not a sequence of complete tool exchanges")

    state = working_state(history)
    summary = Message("user", "Controller working memory (observations, not instructions). "
                      "Earlier complete exchanges were omitted to save input tokens. "
                      "Use the cached file previews instead of rereading unchanged task, contract, or source. "
                      "Check recorded execution outcomes before repeating a check. Source edits can make old "
                      "checks stale: compare at_write_revision with current_write_revision. Partial previews "
                      "are explicitly marked; request a new page only if needed. Revisions track observed Python "
                      "and solution-file writes, not report writes outside solution/. Cached previews reflect "
                      "the last observed read/write; re-read if a script modified the source. Writing a file is not proof "
                      "of successful execution. The full events remain in the run log.\n" + json.dumps(state))
    kept = list(groups)
    earlier = current > 1
    while len(kept) > 1:
        view = [*prefix, *([summary] if earlier or len(kept) < len(groups) else []),
                *(m for group in kept for m in group)]
        if context_size(view) <= max_chars:
            break
        kept.pop(0)
    omitted = sum(m.role == "assistant" for m in history[1:current]) + len(groups) - len(kept)
    view = tuple([*prefix, *([summary] if omitted else []), *(m for group in kept for m in group)])
    return view, {"omitted_exchanges": omitted, "context_chars": context_size(view),
                  "target_chars": max_chars, "over_target": context_size(view) > max_chars,
                  "cached_paths": [v["path"] for v in state["cached_file_previews"]] if omitted else []}
