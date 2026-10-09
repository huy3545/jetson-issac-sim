"""Backend có thể thay bằng Isaac Sim; hiện tại không điều khiển máy thật."""
from typing import Protocol
from safety.safety_monitor import SafetyMonitor

class TargetBackend(Protocol):
    def send_target(self, target: dict) -> dict | None:
        """Gửi target SI; có thể trả feedback x_actual,...,tracking_error."""
        ...

class MockMachineController:
    def __init__(self):
        self.targets = []

    def send_target(self, target: dict) -> None:
        self.targets.append(dict(target))
        return None  # Không tạo feedback giả như measurement thật.

class MachineController:
    def __init__(self, backend: TargetBackend, safety: SafetyMonitor):
        self.backend = backend
        self.safety = safety

    def execute(self, targets):
        """Kiểm tra toàn plan trước khi gửi bất kỳ target nào."""
        result = self.safety.validate(targets)
        if not result.valid:
            raise ValueError(f"Safety rejected: {result.reason}")
        feedback = []
        for target in targets:
            actual = self.backend.send_target(dict(target))
            if actual is not None:
                actual_target = {k: actual.get(k + "_actual") for k in ("x", "y", "z", "c")}
                actual_target["speed"] = target["speed"]
                check = self.safety.validate([actual_target])
                if not check.valid:
                    raise ValueError(f"Feedback rejected: {check.reason}")
            feedback.append(actual)
        return feedback
