"""
examples/solar_panel_example.py
태양광 패널 검사에서의 georeferencing 예시
"""
import numpy as np
import logging
import sys
from pathlib import Path

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.georeferencing.coordinates import GeodeticPoint
from solar_thermal.georeferencing.camera_pose import DronePose, GimbalPose
from solar_thermal.georeferencing.georeferencer import (
    SingleImageGeoreferencer,
    GeoreferencingInput,
    CameraIntrinsics
)


def main():
    logging.basicConfig(level=logging.INFO)
    
    K_rgb = np.array([
        [3550.0,    0.0, 2592.0],
        [   0.0, 3550.0, 1944.0],
        [   0.0,    0.0,    1.0]
    ])
    D_rgb = np.array([-0.12, 0.08, 0.0001, -0.0002, 0.01])
    
    intrinsics = CameraIntrinsics(
        K=K_rgb,
        D=D_rgb,
        image_width=5184,
        image_height=3888
    )
    
    drone_position = GeodeticPoint(
        latitude=37.5665,
        longitude=126.9780,
        altitude=70.2
    )
    
    drone_pose = DronePose(
        yaw_deg=180.0,
        pitch_deg=0.5,
        roll_deg=0.3
    )
    
    gimbal_pose = GimbalPose(
        yaw_deg=180.0,
        pitch_deg=-89.5,
        roll_deg=0.0
    )
    
    PANEL_GROUND_HEIGHT = 50.0
    
    input_data = GeoreferencingInput(
        drone_position=drone_position,
        drone_pose=drone_pose,
        gimbal_pose=gimbal_pose,
        intrinsics=intrinsics,
        ground_height=PANEL_GROUND_HEIGHT,
        ground_normal=np.array([0, 0, 1])
    )
    
    gr = SingleImageGeoreferencer(input_data)
    
    print("=" * 60)
    print("이미지 모서리 (커버리지)")
    print("=" * 60)
    corners = gr.compute_image_corners()
    for i, c in enumerate(["TL", "TR", "BR", "BL"]):
        print(f"  {c}: lat={corners[i].latitude:.7f}, lon={corners[i].longitude:.7f}")
    
    print()
    print("=" * 60)
    print("Ground Sample Distance (GSD)")
    print("=" * 60)
    gsd = gr.compute_ground_sample_distance()
    print(f"  GSD: {gsd*1000:.2f} mm/pixel")
    
    coverage = gr.compute_coverage_area()
    print(f"  커버리지: {coverage:.1f} m² ({coverage/10000:.4f} ha)")
    
    print()
    print("=" * 60)
    print("결함 검출 박스의 지리 좌표")
    print("=" * 60)
    
    detected_bbox = (3000, 1500, 3300, 1700)
    
    center_geo = gr.bbox_center_to_geodetic(detected_bbox)
    print(f"\n  결함 중심:")
    print(f"    위도: {center_geo.latitude:.7f}")
    print(f"    경도: {center_geo.longitude:.7f}")
    
    bbox_corners_geo = gr.bbox_to_geodetic(detected_bbox)
    print(f"\n  결함 박스 4개 모서리:")
    for i, name in enumerate(["TL", "TR", "BR", "BL"]):
        c = bbox_corners_geo[i]
        print(f"    {name}: ({c.latitude:.7f}, {c.longitude:.7f})")
    
    print()
    print("=" * 60)
    print("역변환 검증 (지리 → 픽셀)")
    print("=" * 60)
    
    test_pixel = (3150.0, 1600.0)
    geo = gr.pixel_to_geodetic(test_pixel)
    pixel_back = gr.geodetic_to_pixel(geo)
    
    error = np.sqrt(
        (test_pixel[0] - pixel_back[0]) ** 2 +
        (test_pixel[1] - pixel_back[1]) ** 2
    )
    print(f"  원본 픽셀: {test_pixel}")
    print(f"  역변환 픽셀: ({pixel_back[0]:.2f}, {pixel_back[1]:.2f})")
    print(f"  오차: {error:.4f} px (이상적으로 < 0.001)")


if __name__ == "__main__":
    main()
