# JETSON PROCESSING SIMULATION V1

Module mô phỏng **chức năng phần mềm** dự kiến do Jetson đảm nhiệm trong hệ thống cắt vải/nỉ. Đây không phải Jetson hardware emulator. Chạy trên laptop, không cần Isaac Sim, Jetson, ROS 2, AI hoặc phần cứng điều khiển.

## Cài đặt và chạy

Python 3.10 trở lên:

```bash
cd /home/huy/jetson_sim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
python main.py --scenario straight
python main.py --scenario curve --output-dir output
python -m unittest discover -s tests -v
```

Chỉ dùng NumPy cho hình học/mảng và OpenCV cho ảnh/segmentation. Thinning viết bằng NumPy, không cần scipy, scikit-image hoặc opencv-contrib. Có thể dùng trực tiếp `.venv/bin/python` thay cho `python` khi chưa activate.

## Vision V1.1 — Checkpoint 1

Checkpoint 1 bổ sung các contract và thành phần nền tảng cho Vision V1.1 mà không thay đổi pipeline V1 mặc định.

`python main.py` mặc định chạy `compatibility` mode: giữ nguyên đường xử lý V1 (`VisionProcessor.process(rgb)`), Path Planning, Motion Planning và Machine Controller. Có thể khai báo mode cho thí nghiệm contract mà không thay đổi hành vi controller hiện tại:

```bash
python main.py --mode compatibility
python main.py --mode experimental
python main.py --mode contract-strict
```

Checkpoint này chưa triển khai Safety Supervisor dự báo theo target, reset/replan state machine, thiết kế logging cuối cùng hoặc fault-injection suite hoàn chỉnh.

Các foundation đã có:

- `Config` là immutable (`frozen=True`), có `snapshot()` và `config_version` SHA-256 xác định.
- `Clock`, `FakeClock` và `MonotonicClock` tách clock dữ liệu/mô phỏng khỏi process-health wall clock.
- `MachineState`, `StateSyncBuffer` và `SyncResult` hỗ trợ interpolation, extrapolation giới hạn thực nghiệm và báo rõ `CLOCK_DOMAIN_MISMATCH`.
- `VisionResult` và `LineVisionResult` biểu diễn điểm line, trạng thái quan sát/tracking, confidence, chất lượng depth và nguồn định hướng.
- Semantic Left/Right ưu tiên geometric normal của path, sau đó C-axis, rồi reference; dữ liệu mơ hồ được đánh dấu `AMBIGUOUS`.
- Depth validity được tính sau arc-length resampling, cùng chỉ số với các điểm resample; kết quả gồm ratio và các khoảng chỉ số hợp lệ.
- `LineTracker` duy trì FSM riêng cho từng line: `TRACKING`, `PREDICTING`, `LOST`, dùng delta timestamp và hysteresis.
- `WatchdogCore` là deterministic, phát hiện frame/machine state/vision stale và timestamp frame không tăng; `ProcessHealthAdapter` tách riêng heartbeat tiến trình.

Các ngưỡng bắt đầu bằng `experimental_` trong `VisionConfig` chỉ dành cho mô phỏng/nghiên cứu, không phải thông số an toàn đã xác nhận của máy thật.

### Experimental Vision runtime

`vision.runtime.ExperimentalVisionRuntime` là adapter stateful riêng; nó không thay đổi `VisionProcessor.process(rgb)` hoặc compatibility mode. Adapter nhận `RGBDFrame` có timestamp tăng dần và RGB-D registered, sau đó thực hiện contrast normalization, ROI động, full-frame reacquisition, tracking riêng từng line, confidence, prediction ngắn hạn và hard expiry.

`VisionResult` và `LineVisionResult` phân biệt rõ `OBSERVED`, `OCCLUDED` và `INVALID`. Một predicted line luôn có `verified=False`; khi hết hạn, `points=None`, vì vậy không thể được dùng như cutting geometry hợp lệ. Contract cũng mang `frame_id`, coordinate frame, camera transform, depth validity, missing ranges, contour type, occlusion status và optional fabric-movement uncertainty.

Camera-to-world transform tiếp tục thuộc `CameraCalibration` và Path Planning. Camera-to-cutter/tool transform động cần một transform adapter theo scene/machine state ở giai đoạn tích hợp Isaac Sim, không nằm trong segmentation runtime. Wheel state, occlusion mask và fabric-movement uncertainty hiện là metadata đầu vào; runtime không tự suy ra vải đã trượt nếu không có mask hoặc feedback mô phỏng.

Chạy test Vision runtime experimental:

```bash
python -m unittest tests/test_vision_runtime.py -v
python -m unittest discover -s tests -v
```

### Golden regression V1

Fixture immutable cho kịch bản V1 đường thẳng nằm ở `tests/fixtures/v1_golden_straight.json`; hash được giữ riêng trong `tests/fixtures/v1_golden_straight.sha256`. Fixture có 100 target, snapshot config, metadata môi trường và path-validity. Golden regression kiểm tra integrity SHA-256 trước khi so target với sai số số thực nhỏ.

Môi trường baseline: Python 3.14.4, NumPy 2.5.3 và OpenCV 4.14.0. Dependencies runtime được pin trong `requirements.txt` theo baseline này.

## Kiến trúc và cấu trúc

```text
jetson_sim/
├── config.py
├── clock.py
├── data_source.py
├── machine_state.py
├── watchdog.py
├── vision/vision_processor.py
├── vision/contracts.py
├── vision/tracking.py
├── vision/v11.py
├── path_planning/path_planner.py
├── motion_planning/motion_planner.py
├── controller/machine_controller.py
├── safety/safety_monitor.py
├── logging_data/data_logger.py
├── test_data/
├── tests/
├── output/
├── main.py
├── requirements.txt
└── README.md
```

Mỗi thư mục module có `__init__.py`. Pipeline:

1. **Vision:** chuẩn hóa RGB, HSV segmentation, morphology, nhận hai component, Zhang-Suen thinning, graph tracing 8-neighbor. Trả `left_line`, `right_line` dạng Nx2 pixel `(u,v)`.
2. **Path:** resample theo arc length, đồng bộ chiều hai đường, ghép theo tỷ lệ chiều dài, lấy centerline, moving-average nhẹ giữ endpoint, lấy depth bilinear và chiếu sang tọa độ thế giới. Trả Nx3.
3. **Motion:** tangent, hướng C theo XY, angle unwrapping, curvature 3D, tốc độ theo curvature, backward/forward look-ahead theo giới hạn gia tốc. Speed đầu/cuối bằng 0. Z lấy từ path.
4. **Safety:** path rỗng, NaN/Inf, thiếu field, workspace, khoảng cách waypoint, thay đổi C, speed âm hoặc vượt cấu hình. Trả `VALID` hoặc `INVALID` kèm reason.
5. **Controller:** kiểm tra toàn plan trước khi gọi backend `send_target(target)`. Mock chỉ lưu bản sao target trong bộ nhớ. Khi backend cung cấp feedback, kiểm tra XYZ/C feedback trước khi gửi target kế tiếp.
6. **Logging:** CSV mỗi lần chạy thành công, tên duy nhất, UTC timestamp có timezone. Target và optional feedback tách biệt.

Thứ tự trong `main.py`: Data → Vision → Path → Motion → Safety → Controller → Logging. Mỗi lỗi được báo với module và nguyên nhân, exit code 1; thành công exit code 0. Kế hoạch bị từ chối không được gửi tới backend.

## Dữ liệu, cấu hình và output

`TestDataSource` tạo ảnh hai vạch xanh thẳng/cong cùng depth 0.5 m. Khi chạy main, lưu RGB PNG và depth NPY trong `test_data/`. Đây là dữ liệu tổng hợp, không phải dữ liệu camera hoặc vật liệu thật.

`config.py` chứa HSV, morphology, số waypoint, smoothing, workspace, tốc độ/gia tốc và output directory. Toàn bộ giới hạn chỉ dùng **simulation/test**, không phải thông số an toàn đã kiểm định của máy thật.

Đơn vị: XYZ/depth mét, speed m/s, C radian liên tục, gia tốc m/s². Pixel RGB `(u,v)` tương ứng cột/hàng. Camera dùng quy ước pinhole x sang phải, y xuống dưới, z hướng phía trước; `camera_to_world` biến đổi sang hệ máy/thế giới. `depth` là camera-Z đã đăng ký cùng RGB, không phải khoảng cách Euclidean từ camera.

CSV trong `output/run_*.csv` có:

```text
timestamp,point_id,x_target,y_target,z_target,c_target,speed,path_valid,
x_actual,y_actual,z_actual,c_actual,tracking_error
```

`point_id` bắt đầu từ 0. Feedback hiện trống vì mock không đo chuyển động. Timestamp là thời điểm ghi bản ghi, không phải chu kỳ điều khiển thời gian thực. CSV chỉ được tạo sau khi toàn bộ lệnh hoàn tất; đây chưa phải event log cho các run thất bại.

## Kiểm thử

`python -m unittest discover -s tests -v` chạy V1 end-to-end, CSV, safety, golden regression và các contract Checkpoint 1. Test dùng `unittest` chuẩn, không cần dependency thêm. Các case `INVALID` là kết quả mong đợi và test phải pass.

## PathPlanner V1.1 — Phase 1a

`path_planning/planar_medial_axis.py` là solver XY độc lập cho một kênh hở, sạch, tạo bởi hai polyline biên. Nó không thay đổi `PathPlanner.plan()` V1 và chưa tạo path 3D, không đọc depth, không điều khiển Motion/Controller.

Solver kiểm tra input hữu hạn, số vertex, zero-length segment, tự giao, giao/chạm hai biên; ghép hướng biên theo hình học; dựng polygon kênh; dùng khoảng cách point-to-segment, bisection có bracket để hiệu chỉnh điểm về khoảng cách đều; tái lấy mẫu, chiếu lại và hậu kiểm residual, góc A–P–B, inside-channel và continuity. Kết quả có `SolverResult`, `SolverSegment`, `SolverDiagnostics`, `width_m`, reason code và thống kê số lần gọi distance.

Các ngưỡng Phase 1a nằm trong `PlanarMedialAxisConfig`. Các trường `experimental_*` chỉ là giá trị mô phỏng/nghiên cứu, không phải giới hạn an toàn hoặc thông số máy thật. Solver bảo thủ: khi hậu kiểm không đạt sẽ trả `NOT_CONVERGED` hoặc `INVALID_GEOMETRY`, không trả `CONVERGED` cho điểm chưa xác minh.

Chạy riêng test Phase 1a:

```bash
python -m unittest tests/test_phase1a_preflight.py -v
python -m unittest tests/test_planar_medial_axis.py -v
```

Chạy toàn bộ regression sau thay đổi:

```bash
python -m unittest discover -s tests -v
```

Fixture và threshold khóa nằm tại `tests/fixtures/phase1a_open_channels.json` và `tests/phase1a_thresholds.py`; hash manifest nằm ngoài thư mục test ở `phase1a_fixture_hashes.json`.

### Legacy geometry hardening (experimental)

`path_planning.legacy_geometry_gate.LegacyGeometryGate` là preflight opt-in cho input hai boundary của Legacy `PathPlanner`; nó không thay đổi `PathPlanner.plan()` hay `main.py`, không tự fallback sang Phase 1A và không cấp execution authorization. Gate kiểm tra finite coordinates, point count, degenerate segment, self-intersection của boundary/channel và endpoint correspondence. Nó canonicalize direction trước topology check nên reversed hoặc swapped boundary hợp lệ không bị reject sai.

Kết quả `REJECT` áp dụng cho invalid topology hoặc reference deviation vượt threshold. `UNCERTAIN` áp dụng khi thiếu metric metadata, verified topology/correspondence evidence, reference centerline hoặc reference threshold. `ACCEPT` chỉ có nghĩa là open-channel contract đã được xác minh trong phạm vi gate; không chứng minh Motion/Safety authorization hoặc full-path Phase 1A coverage.

`tests/test_legacy_path_geometry_benchmarks.py` khóa attribution của Legacy theo các bước independent resampling, same-index pairing, midpoint, moving average và RGB-D projection. Tight/S curves với pixel hoàn hảo vẫn được ghi nhận có deviation lớn; test này không làm Legacy V1 im lặng nhận geometry đó là verified.

```bash
.venv/bin/python -m unittest tests/test_legacy_geometry_gate.py tests/test_legacy_path_geometry_benchmarks.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## PathPlanner V1.1 — Phase 1B

`path_planning/vision_adapter.py` là cầu nối experimental từ `VisionResult` sang solver planar Phase 1a. Nó không thay đổi `PathPlanner.plan()` V1, compatibility mode, Motion Planning hoặc Machine Controller.

Adapter chỉ nhận hai boundary đã `OBSERVED`, `TRACKING`, `verified=True`, đủ confidence, không có prediction uncertainty/gap, depth hợp lệ, timestamp hợp lệ và không mơ hồ danh tính. Geometry phải có `units="m"` và coordinate frame tham chiếu đã xác minh; adapter không tự tạo camera-to-cutter transform, và vì thế từ chối input pixel/camera frame. Predicted, expired, lost, gapped, ambiguous, crossing hoặc touching boundary đều trả `REJECTED` kèm reason code.

Closed contour được nhận diện và từ chối rõ ràng với `CLOSED_CONTOUR_UNSUPPORTED` trong Phase 1B hiện tại. Đây là hành vi bảo thủ vì Phase 1a chỉ đã được kiểm chứng cho open channel; không tự mở contour kín tại một seam tùy ý.

Chạy test Phase 1B:

```bash
python -m unittest tests/test_vision_path_adapter.py -v
python -m unittest tests/test_phase1a_preflight.py -v
python -m unittest tests/test_planar_medial_axis.py -v
python -m unittest tests/test_vision_runtime.py -v
python -m unittest discover -s tests -v
python main.py
```

## PathPlanner V1.1 — Phase 2 Surface Mapping

`path_planning.surface_mapping` ánh xạ `GeometrySegment` đã VERIFIED sang XYZ bằng `MetricHeightmap`. Heightmap phải có origin/grid spacing metric, coordinate frame, units `m`, timestamp, clock domain và optional validity mask/uncertainty. Bilinear sampling chỉ tạo điểm VERIFIED khi đủ bốn ô hợp lệ; NaN, invalid mask, out-of-bounds, mismatch frame/time hoặc uncertainty vượt ngưỡng đều bị từ chối.

`SurfaceMapper.map_segments(...)` giữ nguyên source segment ID và không nối qua missing surface samples; thay vào đó trả nhiều `SurfaceSegment` độc lập với `PARTIAL_SURFACE_MAPPING`. Phase 2 không thêm cutting offset, không tính motion, và không tự tạo camera-to-cutter transform. Registered depth vẫn unsupported khi chưa có projection/extrinsic adapter được kiểm chứng.

### Provenance hardening Phase 2 → Phase 3

`MetricHeightmap` là nguồn provenance surface. `SurfaceMappingResult` hiện giữ độc lập provenance của XY (`xy_provenance`) và heightmap (`surface_provenance`); mỗi `SurfaceSegment` giữ stable `segment_id`, `run_id`, `source_indices`, frame, units, timestamp và clock domain có nguồn gốc từ heightmap. Khi mapping bị chia bởi missing samples, mỗi run có input range riêng và không bị nối lại.

Đối số frame/time của mapper và qualifier là **ràng buộc kỳ vọng**: mismatch với metadata nguồn trả diagnostics và không thể tạo `VERIFIED`. `qualify_surface_segments` không dùng metadata caller hay tham số `uncertainty` cũ để nâng cấp kết quả. Uncertainty Z-only từ heightmap vẫn là `UNKNOWN` cho XYZ; strict handoff yêu cầu uncertainty XYZ đã xác minh và threshold được cấu hình. `geometry_only` chỉ dành cho kiểm thử hình học.

Chạy test Phase 2:

```bash
python -m unittest tests/test_surface_mapping.py -v
python -m unittest discover -s tests -v
python main.py
```

## PathPlanner V1.1 — Phase 3 Qualification

`path_planning.path_qualification.qualify_surface_segments(...)` produces immutable, provenance-preserving XYZ handoff segments with 3D arc length, tangent, curvature, uncertainty status and quality gates. Strict mode is default: unknown uncertainty or an unconfigured required uncertainty threshold yields `UNCERTAIN`; `geometry_only` is for experiments only. Phase 3 has no speed, C-axis, acceleration or actuator command.

Qualified arrays are defensive copies marked read-only. `source_id`, child ID, input range, coordinate frame, units, timestamp and clock domain remain in every handoff segment. Invalid source geometry, invalid provenance, duplicate points, step/Z-jump violations or non-finite uncertainty cannot become `VERIFIED`.

Kiểm tra provenance end-to-end:

```bash
.venv/bin/python -m unittest tests/test_surface_mapping.py tests/test_path_qualification.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

```bash
python -m unittest tests/test_path_qualification.py -v
python -m unittest discover -s tests -v
```

## Motion Planning V1.1 — Experimental Adapter Stage 1

`motion_planning.experimental_adapter.ExperimentalMotionAdapter` là entrypoint thử nghiệm nhận `QualifiedPathResult`. Nó không gọi `MotionPlanner` V1, `SafetyMonitor`, `MachineController`, `main.py` hoặc CSV logging. Output là `MotionHandoffResult` gồm `accepted_segments` và `rejected_segments`, mỗi record giữ riêng segment ID, source ID, source index range, XYZ, frame, units, timestamp, clock domain, qualification/uncertainty status, provenance và diagnostics.

Adapter không ghép mảng Nx3 toàn cục, không nối cuối run này với đầu run khác, không thêm waypoint ở gap, và không tạo speed, C-axis, acceleration, actuator command hay execution authorization.

`strict` chỉ nhận segment `VERIFIED`, uncertainty `KNOWN`, provenance metric hợp lệ và không có diagnostic chặn handoff. `geometry_only` có thể nhận geometry chưa đủ measurement uncertainty để nghiên cứu, nhưng record được đánh dấu `experimental=True`, giữ status uncertainty nguồn và luôn `execution_authorized=False`. Chế độ này không thể nâng dữ liệu thành strict handoff.

Chạy test adapter:

```bash
.venv/bin/python -m unittest tests/test_experimental_motion_adapter.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Motion Planning V1.1 — SegmentMotionPlanner Stage 3

`motion_planning.segment_motion_planner.SegmentMotionPlanner` là wrapper experimental gọi lại nguyên `MotionPlanner.plan()` V1 cho **từng** `MotionHandoffSegment` strict. API `plan(MotionHandoffResult) -> SegmentMotionPlanningResult` trả các `planned_segments` và `rejected_segments` riêng biệt; không gộp XYZ, không thêm waypoint ở gap và không tạo transition giữa các run.

`PlannedMotionSegment` giữ segment/source ID, source index range, frame, units, timestamp, clock domain, qualification/uncertainty status, source provenance, diagnostics và target list bất biến. Target là defensive copy của output V1, còn provenance chỉ được giữ ở cấp segment. Kết quả luôn `experimental=True` và `execution_authorized=False`.

Wrapper chỉ nhận `HandoffMode.STRICT`, status `VERIFIED`, uncertainty `KNOWN`, provenance metric hợp lệ và segment không có diagnostic. Geometry-only, unknown uncertainty, INVALID/UNCERTAIN, duplicate, NaN/Inf và XY tangent degenerate đều bị từ chối theo từng run mà không ảnh hưởng run hợp lệ khác.

Motion V1 đặt speed waypoint đầu/cuối bằng `0.0`; Stage 3 kiểm tra số học policy này sau mỗi lần gọi V1, đồng thời kiểm tra target count, finite fields, speed không âm và XYZ output khớp input. Nếu các kiểm tra này thất bại, segment bị reject; wrapper không tự sửa speed hay thêm waypoint.

```bash
.venv/bin/python -m unittest tests/test_segment_motion_planner.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Motion Planning V1.1 — Quality Regression Stage 5A

`tests/test_motion_quality_regression.py` là baseline numerical xác định cho Motion V1/V1.1. Test tạo trực tiếp đường thẳng, cung R=1 m và R=0.05 m, S-curve, lấy mẫu không đều, cung đổi Z, cung có nhiễu Gaussian 0.3 mm với seed cố định, segment hai waypoint, XY gần suy biến/chỉ đổi Z, và hai run có gap.

Suite kiểm tra arc length, tangent/C-axis giải tích trên cung, unwrap, curvature nội bộ bằng estimator circumcircle độc lập, sai lệch endpoint của finite difference, speed cap, lateral cap với tolerance số học `2e-6 m/s²`, ràng buộc gia tốc rời rạc, finite targets, boundary speed và provenance/run isolation. Đây là regression numerical, không phải physical validation hay trajectory timing. Không thêm smoothing, resampling, timing hoặc jerk limiting vào Motion V1.

```bash
.venv/bin/python -m unittest tests/test_motion_quality_regression.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Path Planning V1.1 — Experimental Geometry-Gated Entrypoint Stage 5F

`path_planning.experimental_geometry_entrypoint.ExperimentalGeometryEntrypoint` is an opt-in research entrypoint around the unchanged Legacy `PathPlanner`. It evaluates `LegacyGeometryGate` on metric boundaries first. Only a gate result of `ACCEPT` can invoke the Legacy benchmark; `REJECT` and `UNCERTAIN` return their diagnostics with `benchmark_status=NOT_RUN` and never call the planner.

The benchmark requires a separate, explicit `LegacyBenchmarkContext`: pixel boundaries, finite positive registered depth, `CameraCalibration`, matching source/timestamp/clock-domain metadata, and declared mapping provenance. The entrypoint never infers a pixel-to-metric transform, calibration, uncertainty, or evidence. Synthetic fixtures must set `synthetic=True` and still declare the mapping provenance.

`ExperimentalGeometryResult` preserves the gate result, source metadata snapshots and diagnostics. Its output path is a defensive read-only copy. It is always marked `experimental=True` and `execution_authorized=False`; an `ACCEPT` therefore permits only a research benchmark, never machine execution. It imports no Phase 1A/1B fallback, Motion, Safety, Controller, `main.py`, or CSV logging.

```bash
.venv/bin/python -m unittest tests/test_experimental_geometry_entrypoint.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Vision V1.1 — Experimental Metric Projection Stage 6B

`path_planning.experimental_metric_projection.ExperimentalVisionMetricProjectionAdapter` projects two admissible Vision V1.1 pixel boundaries to reference-frame XYZ without modifying the Legacy `PathPlanner`. It requires `TransformEvidence`: named camera/reference frames, calibration and transform IDs/versions/sources/timestamps, source identity, clock domain, units, a rigid 4×4 transform, and an explicit validity label. The adapter validates the homogeneous row, finite translation, SO(3) orthonormality and determinant `+1` (default tolerance `1e-8`). Reflection, scale, shear and invalid homogeneous transforms are rejected.

Admission requires both lines to be `OBSERVED`, `TRACKING`, Vision-verified, finite, non-ambiguous and gap-free with a true depth-valid mask for every original sample. It preserves those samples without resampling, rejects invalid/zero/missing registered depth and never extrapolates outside the depth image. Vision frame/timestamp/domain/frame ID and `camera_to_reference` must match the supplied evidence exactly.

Projection uses the existing pinhole equation and bilinear registered-depth semantics. Numerical success creates immutable XYZ segments but always returns `UNCERTAIN` with `UNKNOWN` uncertainty in Stage 6B. A synthetic or caller-labelled `VERIFIED` transform is not independent calibration evidence, so the adapter never emits metric `VERIFIED`, `KNOWN` uncertainty or execution authorization.

```bash
.venv/bin/python -m unittest tests/test_experimental_metric_projection.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Vision V1.1 — Calibration Evidence & Trust Stage 6C

`path_planning.calibration_evidence.CalibrationEvidence` records calibration identity, camera/frame metadata, intrinsic/extrinsic references, source/method/timestamp, optional artifact reference/digest and validation metadata. `CalibrationEvidenceValidator` returns one of `SYNTHETIC`, `DECLARED`, `EVIDENCE_VALIDATED` or `INVALID`. `EVIDENCE_VALIDATED` proves only that supplied artifact bytes match the supplied digest and validation metadata is complete; it does not prove physical calibration accuracy.

`CalibrationTrustedProjectionAdapter` is an opt-in wrapper around Stage 6B. It checks calibration evidence against `TransformEvidence` before calling the unchanged projection adapter. Missing IDs, malformed/mismatched artifact digests, failed validation claims, frame/unit/clock/source or intrinsic/extrinsic-reference mismatches return `INVALID` and do not produce a metric result. Caller metadata alone remains `DECLARED`, even if it self-labels as validated.

Every permitted projection remains experimental, non-executable and `UNCERTAIN` with `UNKNOWN` XYZ uncertainty. Synthetic evidence cannot be mixed with a non-synthetic transform. The module does not read artifacts, infer evidence, call Legacy PathPlanner, Motion, Safety, Controller or logging.

```bash
.venv/bin/python -m unittest tests/test_calibration_evidence.py -v
.venv/bin/python -m unittest tests/test_experimental_metric_projection.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Vision V1.1 — Experimental XYZ Uncertainty Stage 6D

`path_planning.xyz_uncertainty.ExperimentalXYZUncertaintyPropagator` is an opt-in layer after Stage 6B/6C admission. `UncertaintyEvidence` defines covariance ordering: pixel `[u,v]`, intrinsics `[fx,fy,cx,cy]`, registered-depth neighbourhood `[z00,z10,z01,z11]`, and extrinsics `[tx,ty,tz,rx,ry,rz]`. Block-diagonal propagation is permitted only when `zero_cross_covariance_confirmed=True`; correlated depth samples use `wᵀΣdepthw` with actual bilinear weights.

It uses the analytic pinhole Jacobian and a left/reference-frame SE(3) perturbation convention: `T'=Exp([δt,δθ])T`, `δp_ref=δt+δθ×p_ref`, with extrinsic Jacobian `[I,-skew(p_ref)]`. All covariance inputs must be finite, symmetric and PSD. Invalid or missing covariance is rejected; it is never replaced by zero.

Only synthetic evidence produces `MODELLED_SYNTHETIC` covariance. This numerical model still returns geometry `UNCERTAIN`, uncertainty `UNKNOWN`, `experimental=True` and `execution_authorized=False`; it is not calibrated sensor uncertainty and cannot enable strict planning.

```bash
.venv/bin/python -m unittest tests/test_xyz_uncertainty.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Vision V1.1 — Uncertainty Evidence Ingestion Stage 6F

`path_planning.uncertainty_artifact.UncertaintyArtifactIngestor` accepts bounded raw JSON bytes and validates a versioned uncertainty artifact without reading artifact paths. SHA-256 is calculated over the exact original bytes before parsing. A matching caller-provided digest yields `EVIDENCE_VALIDATED`; valid metadata without a digest remains `DECLARED`. Neither state proves empirical sensor accuracy.

The artifact carries fixed Stage 6D variable ordering and PSD covariance blocks for pixel, registered depth, intrinsics and extrinsics. It also preserves correlation metadata. Only `EXPLICIT_ZERO_ASSUMPTION` with a named source maps to `UncertaintyEvidence`; `UNKNOWN` is not propagated and `PROVIDED_CROSS_COVARIANCE` is retained but marked unsupported because Stage 6D has no cross-block propagation. Every result remains experimental and non-executable.

```bash
.venv/bin/python -m unittest tests/test_uncertainty_artifact.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Vision V1.1 — Experimental Full Covariance Stage 6H

`path_planning.full_covariance` adds a separate `1.1-full-covariance` artifact schema with canonical 16-variable ordering `[u,v,z00,z10,z01,z11,fx,fy,cx,cy,tx,ty,tz,rx,ry,rz]`. A full 16×16 matrix must be finite, symmetric and PSD; named Stage 6D blocks must exactly match its diagonal submatrices. No eigenvalue clipping, symmetrisation or implicit zero cross-block fill is performed.

`ExperimentalFullCovariancePropagator` uses `JΣfullJᵀ` once with a 3×16 Jacobian assembled from the Stage 6D pinhole, bilinear-depth and left-perturbation extrinsic Jacobians. It accepts only `PROVIDED_CROSS_COVARIANCE` artifacts with a numerical full matrix and correlation provenance. Output remains immutable, experimental, non-executable, `UNCERTAIN` and `UNKNOWN`; valid structure or digest never proves calibrated sensor uncertainty.

```bash
.venv/bin/python -m unittest tests/test_full_covariance.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

## Vision V1.1 — Experimental Cross-Point Covariance Stage 6J

`path_planning.cross_point_covariance` adds the `1.2-cross-point` artifact schema for small point sets. Its global ordering is each point's `[u,v,z00,z10,z01,z11]` followed once by shared `[fx,fy,cx,cy,tx,ty,tz,rx,ry,rz]`. Point IDs/order, frame/time/domain, calibration/transform IDs and shared uncertainty identity are explicit; matching numerical values never imply shared provenance.

The global covariance has shape `(6N+10)²`; propagation uses one global Jacobian `(3N)×(6N+10)` and preserves all diagonal and off-diagonal 3×3 XYZ blocks. Dense processing is limited to 32 points (maximum input covariance 202×202, about 0.33 MB float64) and oversized requests are rejected before NumPy covariance allocation. Results remain immutable, experimental, `UNCERTAIN`, `UNKNOWN` and non-executable.

```bash
.venv/bin/python -m unittest tests/test_cross_point_covariance.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```

### Advanced & Partial Geometry

`path_planning.advanced_geometry.AdvancedGeometryPlanner` là API experimental bổ sung cho Phase 1B. `plan_partial(left, right, left_valid, right_valid)` tạo các `GeometrySegment` độc lập theo từng run quan sát hợp lệ; nó không nội suy hoặc nối VERIFIED geometry qua gap. Segment mang `VERIFIED`, `INVALID` hoặc `UNCERTAIN`, input range và diagnostics.

`plan_closed_concentric(inner, outer)` chỉ xác minh annulus tạo bởi hai vòng tròn đồng tâm, kín và không giao nhau. Kết quả không phụ thuộc cyclic starting index. Closed contour không đồng tâm, self-intersection hoặc topology chưa chứng minh được vẫn trả `UNCERTAIN`/`INVALID`; Phase 1B chưa hỗ trợ mọi closed contour hoặc tự chọn seam.

Chạy test Advanced/Partial Geometry:

```bash
python -m unittest tests/test_advanced_geometry.py -v
python -m unittest discover -s tests -v
python main.py
```

## Tích hợp Isaac Sim ở V2

Implement adapter thực tế khi chuyển sang máy chạy Isaac Sim 5.1.0:

- `IsaacSimRGBDDataSource.read() -> RGBDFrame`, theo `RGBDDataSource` protocol: RGB uint8 HxWx3 hoặc float [0,1], depth mét HxW đồng bộ/registered với RGB, intrinsics và `camera_to_world` 4x4 từ camera thật trong scene. Chuyển đổi đúng hệ trục và đơn vị scene sang hợp đồng SI này.
- `IsaacSimMachineController.send_target(target) -> dict | None`, theo `TargetBackend` protocol: chuyển XYZ/C sang cơ cấu trong scene; feedback có `x_actual`, `y_actual`, `z_actual`, `c_actual`, `tracking_error` nếu có. C feedback cần cùng quy ước góc unwrapped.
- Truyền source/backend qua `run(source=..., backend=..., config=...)`. Năm module xử lý, safety, logging tiếp tục dùng chung.

V1 không chứa API Isaac Sim giả hoặc import simulator. Adapter V2 cần bổ sung lịch step/synchronization, polling feedback theo thời gian, theo dõi sai số, stop/cancel backend và semantics tốc độ trước khi điều khiển scene. `send_target` hiện là contract đồng bộ; speed là feedrate target, chưa phải trajectory định thời hoặc mô hình motor.

Giới hạn V1: yêu cầu hai đường xanh tách rời, không xử lý nhánh/giao nhau/tắc khuất; correspondence theo arc length không giải quyết đường có chiều dài khác biệt lớn; moving-average là smoothing nhẹ, chưa bảo đảm đường cắt nằm trong corridor; không có collision detection hay kiểm chứng động lực học của máy. Camera calibration và offset dụng cụ cần xác định ở V2.

## Stage 7B — Experimental Integration Contract & Observability

`integration.experimental_harness.ExperimentalIntegrationHarness` là entrypoint nghiên cứu độc lập với `main.py`. Nó tạo các `IntegrationRecord` bất biến cùng `IntegrationEvent` cho Vision, Path, bridge Vision→Phase 3, Motion và Safety assessment. Mọi record giữ `run_id`, `correlation_id`, stage, frame/time/frame-domain, coordinate frame, units, config version, provenance references, confidence, geometry/uncertainty/qualification status và reason codes khi dữ liệu nguồn có chúng.

Harness gọi `VisionPathAdapter`, `ExperimentalMotionAdapter` và `SegmentMotionPlanner` theo public contract hiện có. Nó không tạo geometry thay thế cho dữ liệu bị từ chối và đánh dấu bridge Vision→Phase 3 là `BLOCKED` vì repository chưa có contract chứng minh `VisionPathResult` và `QualifiedPathResult` cùng source run. `ExperimentalSafetyAssessment` là đánh giá fail-closed độc lập, không thay đổi `SafetyMonitor` V1, luôn từ chối execution ở Stage 7B. Module không import hoặc gọi `MachineController`, hardware backend hay CSV logger V1.

Structured audit logging mặc định là in-memory và passive: lỗi sink chỉ đặt `audit_complete=False` cùng event `AUDIT_LOGGING_FAILED`; không được biến quyết định stage thành thành công. Mọi kết quả Stage 7B vẫn có `execution_authorized=False`.

```bash
.venv/bin/python -m unittest tests/test_experimental_integration.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```

## Stage 7B-1 — Versioned Contracts & Point-Level Provenance

`integration.contracts` là nguồn định nghĩa duy nhất cho schema experimental `1.1`. `IntegrationRecord`, `IntegrationEvent` và `IntegrationRunResult` mang `schema_version`; schema chưa hỗ trợ, enum không thuộc contract, identifier sai định dạng, số không hữu hạn và `execution_authorized=True` đều bị từ chối khi khởi tạo. Trường thiếu vẫn có thể được biểu diễn để ghi nhận failure, nhưng strict admission phải trả reason code thay vì tự điền giá trị.

`PointProvenance` bất biến giữ `point_id`, toàn bộ `source_point_ids`, run/frame/segment identity, observation/confidence/uncertainty status, frame/time, calibration/transform identity và `derivation_method`. Các phương thức như interpolation, resampling, smoothing, midpoint hoặc coordinate transform phải giữ references đến mọi điểm nguồn. Strict policy từ chối `PREDICTED`, confidence `UNKNOWN`, uncertainty không `KNOWN`, hoặc metadata bắt buộc thiếu; không có trust upgrade ngầm.

`MotionTimingRecord` mô tả target order, segment/path identity, frame/units, speed và C-axis. Khi thiếu timestamp lẫn duration, nó trả `TIMING_UNSPECIFIED`; Stage 7B-1 không suy ra thời gian từ speed. `SyntheticConfiguration` buộc camera/calibration/transform simulation có identity rõ ràng, luôn `real_accuracy_claim="NONE"`. Configuration thực phải là `UNSPECIFIED` và fail closed; tolerance synthetic không phải dung sai máy thật.

```bash
.venv/bin/python -m unittest tests/test_integration_contract_hardening.py -v
.venv/bin/python -m unittest tests/test_experimental_integration.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```

## Stage 7B-1A — Contract Validation Fixes

`integration.validation.ProvenanceResolver` là strict validator đệ quy cho `PointProvenance`. Điểm quan sát gốc dùng `derivation_method=OBSERVATION` và không có parent giả. Điểm derived phải giải quyết được toàn bộ `source_point_ids`; source thiếu trả `SOURCE_PROVENANCE_UNRESOLVED`, source `PREDICTED`/`UNKNOWN` bị từ chối, run không khớp trả `SOURCE_RUN_MISMATCH` và cycle trả `PROVENANCE_CYCLE`. Validator không tin riêng nhãn của điểm đầu ra và không thay đổi provenance để hợp thức hóa admission.

`CompatibilityValidator` kiểm tra run/frame/time, clock domain, coordinate frame, units, calibration/transform, config version và segment identity. Timestamp khác nhau chỉ được chấp nhận khi caller cung cấp chính sách đồng bộ hữu hạn rõ ràng; không có tolerance thì trả `FRAME_TIME_MISMATCH`. `PointProvenance`, motion timing và integration records cũng từ chối bool tại numeric fields, tuple nested sai cấu trúc, enum/schema không hỗ trợ và execution authorization.

```bash
.venv/bin/python -m unittest tests/test_integration_validation.py -v
.venv/bin/python -m unittest tests/test_integration_contract_hardening.py -v
.venv/bin/python -m unittest tests/test_experimental_integration.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```

## Stage 7B-1b — Remaining Contract Integrity

Stage 7B-1b siết integrity mà không thay solver hoặc Legacy V1. `ProvenanceResolver` yêu cầu dictionary key đúng bằng `PointProvenance.point_id`, kiểm tra compatibility của từng cạnh trong lineage và hỗ trợ timestamp tolerance chỉ khi caller khai báo chính sách sync hữu hạn. `ProvenanceMerger` là helper production cho interpolation, resampling, smoothing, midpoint và coordinate transform: nó giữ toàn bộ source references, chặn source predicted/unknown hoặc metadata không tương thích, và tạo derived point với confidence/uncertainty `UNKNOWN` thay vì tự nâng trust.

Contracts từ chối `ACCEPTED` có rejection reason, bool tại identifier/numeric fields, uncertainty enum lạ, nested run/event sai run-correlation và final status trái với record `BLOCKED`. Motion timestamp đơn lẻ không chứng minh timing readiness; khi không có duration, `MotionTimingRecord` trả `TIMING_UNSPECIFIED`.

Không có mutation framework dependency trong repository. Test `test_contract_integrity.py` dùng mutation cases độc lập: cố tạo accepted record có rejection reason và resolver registry có key khác `point_id`; cả hai phải bị production guard từ chối.

```bash
.venv/bin/python -m unittest tests/test_contract_integrity.py -v
.venv/bin/python -m unittest tests/test_integration_validation.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```

## Stage 7B-2 — Independent SafetyDecision & Immutable Plans

`integration.independent_safety` thêm `PathPlan` và `MotionPlan` immutable cho nghiên cứu. Mảng XYZ được defensive-copy thành read-only; `content_hash` là SHA-256 trên canonical serialization gồm coordinates và metadata quan trọng, không phải Python `hash()` hoặc chỉ ID. `MotionPlan.source_path_hash` phải khớp PathPlan thực tế, mỗi target giữ reference tới point path, và timing thiếu duration trả `TIMING_UNSPECIFIED`.

`IndependentSafetyValidator` không dùng `SafetyMonitor` V1, không tin planner status và không import Controller. Nó tự kiểm tra run/correlation, path hash, lineage provenance, calibration/transform, qualification/uncertainty, gaps, frame/domain/units, target order, timing và giới hạn `SyntheticMachineLimits`. Mọi `SafetyDecision` đều `execution_authorized=False`; `ACCEPTED` chỉ nghĩa là đạt kiểm tra mô phỏng Stage 7B-2, không phải quyền điều khiển máy.

`apply_boundary_veto` chặn downstream khi upstream hoặc SafetyDecision không `ACCEPTED`. Audit sink là passive; lỗi sink chuyển decision thành `BLOCKED` với `EVIDENCE_INCOMPLETE`, không cấp authorization. Stage này chưa mở Vision→Phase 3 bridge, không gọi MachineController và không triển khai execution mock.

```bash
.venv/bin/python -m unittest tests/test_independent_safety.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```

## Stage 7B-2A — Independent Safety Boundary Hardening

`SyntheticMachineState` thay thế evidence machine-state kiểu tự do. Contract immutable này bắt buộc run/correlation/frame/time/clock domain, configuration identity, giá trị simulation hữu hạn, evidence status và `real_accuracy_claim="NONE"`. Object rỗng hoặc arbitrary object không còn được Safety chấp nhận là evidence.

`apply_boundary_veto(..., protected=True, path=..., motion=..., limits=...)` bắt buộc `SafetyDecision` còn phù hợp tại boundary Safety→Mock Control: decision phải `ACCEPTED` và khớp run/correlation, path/motion IDs và hashes, cùng safety config version. Thiếu, stale hoặc unrelated decision đều `BLOCKED`. Boundary không được safety-protected vẫn có thể hoạt động với upstream acceptance theo contract cũ.

`PathPlan` và `MotionPlan` lưu NumPy XYZ trong immutable bytes-backed buffers. Caller array, direct attribute, view và thử `setflags(write=True)` không thể sửa plan hoặc làm hash không còn đúng. Safety cũng kiểm tra metadata point/path, timing target index/path/segment, provenance reference, plan config và machine-state identity/domain/configuration.

```bash
.venv/bin/python -m unittest tests/test_safety_boundary_hardening.py -v
.venv/bin/python -m unittest tests/test_independent_safety.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```

## Stage 7B-3 — FSM, Coordinator & Structured Audit

`integration.coordinator.IntegrationCoordinator` là entrypoint experimental độc lập với `main.py`. FSM dùng bảng transition cố định: `IDLE → SCAN → PREVIEW → CONFIRM → CUT`; lỗi evidence chuyển `HOLD`, và `HOLD`/`FAULT` chỉ về `IDLE` qua reset có audit. `CUT` chỉ là trạng thái logic mô phỏng, không phát target, không gọi Controller và `execution_authorized` luôn `False`.

Boundary registry cố định gồm `vision-path`, `path-motion`, `motion-safety` và `safety-simulation-cut`; boundary cuối luôn protected. Coordinator dùng `IndependentSafetyValidator` và `apply_boundary_veto`, không tự tạo SafetyDecision. Timestamp khác chỉ được chấp nhận khi có `SynchronizationPolicy` versioned; nếu không có bằng chứng sync, kết quả vào `HOLD` với `MACHINE_STATE_TIME_UNVERIFIED`.

`CoordinatorAuditEvent` giữ thứ tự deterministic, FSM states, contract hashes, safety reference, reason codes và audit completeness. Audit sink failure dẫn đến `HOLD` với evidence incomplete, không crash Legacy V1.

```bash
.venv/bin/python -m unittest tests/test_integration_coordinator.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py
```
# jetson-issac-sim
