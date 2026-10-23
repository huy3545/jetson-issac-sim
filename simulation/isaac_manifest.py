"""Loader and validator for Isaac Sim canonical scene manifest."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

MANIFEST_SCHEMA_VERSION = "1.0.0"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "configs" / "isaac_scene_manifest.json"


@dataclass(frozen=True)
class IsaacSceneManifest:
    schema_version: str
    description: str
    target_scene: MappingProxyType
    prim_paths: MappingProxyType
    kinematics: MappingProxyType
    safety: MappingProxyType

    @property
    def default_stage_file(self) -> str:
        return self.target_scene["default_stage_file"]

    @property
    def machine_root(self) -> str:
        return self.prim_paths["machine_root"]

    @property
    def legacy_aliases(self) -> tuple:
        return tuple(self.prim_paths.get("legacy_aliases", ()))

    @property
    def preview_namespace(self) -> str:
        return self.prim_paths["preview_namespace"]

    @property
    def preview_prims(self) -> MappingProxyType:
        return self.prim_paths["preview_prims"]

    @property
    def execution_authorized(self) -> bool:
        return bool(self.safety.get("execution_authorized", False))

    @property
    def gate_b_status(self) -> str:
        return str(self.safety.get("gate_b_status", "BLOCKED"))

    @property
    def kinematics_evidence_status(self) -> str:
        return str(self.kinematics.get("kinematics_evidence_status", "UNVERIFIED"))

    @property
    def camera_verification_status(self) -> str:
        sensor_info = self.prim_paths.get("sensors", {})
        camera_sensor = sensor_info.get("camera_sensor", {})
        if isinstance(camera_sensor, dict):
            return str(camera_sensor.get("verification_status", "UNVERIFIED"))
        return "UNVERIFIED"

    @property
    def depth_evidence_status(self) -> str:
        sensor_info = self.prim_paths.get("sensors", {})
        camera_sensor = sensor_info.get("camera_sensor", {})
        if isinstance(camera_sensor, dict):
            return str(camera_sensor.get("depth_evidence_status", "NOT_VERIFIED"))
        return "NOT_VERIFIED"

    def get_camera_path(self) -> str:
        """Return canonical camera Prim path with backward compatibility."""
        sensors = self.prim_paths.get("sensors", {})
        if "camera_rgbd" in sensors and isinstance(sensors["camera_rgbd"], str):
            return sensors["camera_rgbd"]
        camera_sensor = sensors.get("camera_sensor", {})
        if isinstance(camera_sensor, dict) and "canonical_path" in camera_sensor:
            return camera_sensor["canonical_path"]
        return "/World/AirTable/Camera_Sensor"

    def get_candidate_paths(self, component: str) -> tuple:
        """Retrieve candidate Prim paths and origin scene evidence for a component."""
        if component in ("camera", "camera_sensor", "camera_rgbd"):
            sensor = self.prim_paths.get("sensors", {}).get("camera_sensor", {})
            return tuple(sensor.get("candidate_paths", ())) if isinstance(sensor, dict) else ()

        actuator_details = self.prim_paths.get("actuators", {}).get("details", {})
        if isinstance(actuator_details, dict) and component in actuator_details:
            entry = actuator_details[component]
            if isinstance(entry, dict):
                return tuple(entry.get("candidate_paths", ()))

        return ()


def load_manifest(manifest_path: Path | str | None = None) -> IsaacSceneManifest:
    """Load and validate the Isaac Sim scene manifest."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Isaac scene manifest not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid manifest schema_version: expected {MANIFEST_SCHEMA_VERSION}, got {data.get('schema_version')}"
        )

    required_sections = ("target_scene", "prim_paths", "fabric_differential_kinematics", "safety_enforcement")
    for section in required_sections:
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"Manifest missing required dictionary section: '{section}'")

    # Safety enforcement check: execution_authorized must be explicitly false for preview
    safety = data["safety_enforcement"]
    if safety.get("execution_authorized") is not False:
        raise ValueError("Safety violation: manifest execution_authorized must be False")

    return IsaacSceneManifest(
        schema_version=data["schema_version"],
        description=data.get("description", ""),
        target_scene=MappingProxyType(dict(data["target_scene"])),
        prim_paths=MappingProxyType(dict(data["prim_paths"])),
        kinematics=MappingProxyType(dict(data["fabric_differential_kinematics"])),
        safety=MappingProxyType(dict(safety)),
    )
