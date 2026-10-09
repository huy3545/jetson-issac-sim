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
# jetson-issac-sim
