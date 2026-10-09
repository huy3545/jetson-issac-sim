"""Chạy JETSON PROCESSING SIMULATION V1 độc lập Isaac Sim."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from config import Config
from data_source import TestDataSource
from vision.vision_processor import VisionProcessor
from path_planning.path_planner import PathPlanner
from motion_planning.motion_planner import MotionPlanner
from safety.safety_monitor import SafetyMonitor
from controller.machine_controller import MachineController, MockMachineController
from logging_data.data_logger import DataLogger

def run(scenario="curve", config=None, source=None, backend=None):
    cfg = config or Config()
    stage = "Load/Create Test Data"
    print("=" * 46 + "\nJETSON PROCESSING SIMULATION V1\n" + "=" * 46)
    try:
        frame = (source or TestDataSource(scenario)).read()
        if frame.depth.shape != frame.rgb.shape[:2]:
            raise ValueError("RGB và depth phải có cùng kích thước pixel")
        if source is None:
            folder = Path(__file__).resolve().parent / "test_data"
            folder.mkdir(exist_ok=True)
            if not cv2.imwrite(str(folder / f"{scenario}_rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                raise OSError("Không lưu được test RGB")
            np.save(folder / f"{scenario}_depth.npy", frame.depth)
        stage = "Vision Processing"
        print("\n[1] " + stage)
        left, right = VisionProcessor(cfg.vision).process(frame.rgb)
        print("Status: OK")
        stage = "Path Planning"
        print("\n[2] " + stage)
        path = PathPlanner(cfg.path).plan(left, right, frame.depth, frame.calibration)
        print(f"Status: OK\nPath Points: {len(path)}")
        stage = "Motion Planning"
        print("\n[3] " + stage)
        targets = MotionPlanner(cfg.motion).plan(path)
        print("Status: OK")
        stage = "Safety / Monitoring"
        print("\n[4] " + stage)
        safety = SafetyMonitor(cfg.safety)
        validation = safety.validate(targets)
        print(f"Status: {validation.status}\nReason: {validation.reason}")
        if not validation.valid:
            raise ValueError(validation.reason)
        stage = "Machine Control"
        print("\n[5] " + stage)
        feedback = MachineController(backend or MockMachineController(), safety).execute(targets)
        print(f"Targets Generated: {len(targets)}")
        stage = "Data Logging"
        print("\n[6] " + stage)
        output = DataLogger(cfg.output_dir).save(targets, validation.valid, feedback)
        print(f"Saved: {output}")
        print("\n" + "=" * 46 + "\nJETSON_SIM RESULT: SUCCESS\n" + "=" * 46)
        return output
    except Exception as error:
        print(f"\nModule: {stage}\nStatus: ERROR\nReason: {error}\nJETSON_SIM RESULT: FAILED")
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["straight", "curve"], default="curve")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--mode", choices=["compatibility", "experimental", "contract-strict"], default="compatibility")
    args = parser.parse_args()
    cfg = Config(output_dir=args.output_dir, mode=args.mode)
    return 0 if run(args.scenario, cfg) is not None else 1

if __name__ == "__main__":
    raise SystemExit(main())
