def run_policy(mode, runner, max_revisions=1):
    if mode not in {"single", "multi"}:
        raise ValueError("Mode must be single or multi")
    handoffs = []
    stage = "single" if mode == "single" else "diagnosis"
    revisions = 0
    transitions = {"diagnosis": "features", "features": "modeling", "modeling": "auditing"}
    while True:
        result = runner.run_stage(stage, handoffs)
        handoffs.append(result)
        if result["status"] == "rejected":
            raise RuntimeError(f"Stage {stage} rejected the candidate")
        if result["status"] == "revise":
            revisions += 1
            if revisions > max_revisions:
                raise RuntimeError("Revision limit exhausted; candidate not accepted")
            target = "single" if mode == "single" else result["next_role"]
            allowance = getattr(runner, "allowances", {}).get(target)
            if allowance is not None and allowance.finalizing:
                raise RuntimeError(f"Revision target {target} is already finish-only; code work cannot reopen")
            stage = target
            continue
        if stage in {"single", "auditing"}:
            return handoffs
        stage = transitions[stage]
