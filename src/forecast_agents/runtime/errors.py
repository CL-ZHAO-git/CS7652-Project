"""Local execution failures whose raw output must not enter model feedback."""


class PredictionContractError(ValueError):
    """Controller-authored fixed messages without candidate values or raw rows."""


class CandidateExecutionError(RuntimeError):
    def __init__(self, operation, execution):
        self.operation = operation
        self.execution = execution
        super().__init__(f"Candidate {operation} execution failed; details are stored locally")
