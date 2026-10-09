"""CSV target và feedback tùy chọn; mỗi run dùng tên duy nhất."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

class DataLogger:
    fields = ["timestamp", "point_id", "x_target", "y_target", "z_target", "c_target", "speed", "path_valid",
              "x_actual", "y_actual", "z_actual", "c_actual", "tracking_error"]

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def save(self, targets, path_valid: bool, feedback=None) -> Path:
        if feedback is not None and len(feedback) != len(targets):
            raise ValueError("Số feedback không khớp số target")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"run_{stamp}_{uuid4().hex[:8]}.csv"
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            for i, target in enumerate(targets):
                row = dict(timestamp=datetime.now(timezone.utc).isoformat(), point_id=i, speed=target["speed"], path_valid=path_valid)
                row.update({f"{k}_target": target[k] for k in ("x", "y", "z", "c")})
                actual = feedback[i] if feedback is not None else None
                row.update({k: (actual or {}).get(k) for k in self.fields[8:]})
                writer.writerow(row)
        return path
