"""
tests/test_georeferencer.py
"""
import numpy as np
import pytest
import sys

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.georeferencing.coordinates import (
    GeodeticPoint, ENUPoint,
    geodetic_to_ecef, ecef_to_geodetic,
    geodetic_to_enu, enu_to_geodetic
)
from solar_thermal.georeferencing.camera_pose import DronePose, GimbalPose, euler_to_rotation_matrix
from solar_thermal.georeferencing.plane_intersection import Plane, Ray, intersect_ray_plane
from solar_thermal.georeferencing.georeferencer import (
    SingleImageGeoreferencer, GeoreferencingInput, CameraIntrinsics
)


class TestCoordinateConversions:
    
    def test_geodetic_ecef_roundtrip(self):
        """지리 ↔ ECEF 변환의 역변환 정확도"""
        original = GeodeticPoint(
            latitude=37.5665,
            longitude=126.9780,
            altitude=50.0
        )
        
        ecef = geodetic_to_ecef(original)
        recovered = ecef_to_geodetic(ecef)
        
        assert abs(original.latitude - recovered.latitude) < 1e-9
        assert abs(original.longitude - recovered.longitude) < 1e-9
        assert abs(original.altitude - recovered.altitude) < 1e-3
    
    def test_geodetic_enu_roundtrip(self):
        """지리 ↔ ENU 변환의 역변환"""
        origin = GeodeticPoint(37.5665, 126.9780, 50.0)
        original = GeodeticPoint(37.5670, 126.9785, 55.0)
        
        enu = geodetic_to_enu(original, origin)
        recovered = enu_to_geodetic(enu, origin)
        
        assert abs(original.latitude - recovered.latitude) < 1e-9
        assert abs(original.longitude - recovered.longitude) < 1e-9
        assert abs(original.altitude - recovered.altitude) < 1e-3
    
    def test_enu_origin_is_zero(self):
        """원점 자체의 ENU는 (0, 0, 0)"""
        origin = GeodeticPoint(37.5665, 126.9780, 50.0)
        enu = geodetic_to_enu(origin, origin)
        
        assert abs(enu.east) < 1e-6
        assert abs(enu.north) < 1e-6
        assert abs(enu.up) < 1e-6


class TestPlaneIntersection:
    
    def test_vertical_ray_to_horizontal_plane(self):
        """수직 광선이 수평 평면에 정확히 만남"""
        ray = Ray(
            origin=np.array([10.0, 20.0, 100.0]),
            direction=np.array([0.0, 0.0, -1.0])
        )
        plane = Plane.horizontal_at_height(0.0)
        
        intersection = intersect_ray_plane(ray, plane)
        
        assert intersection is not None
        np.testing.assert_array_almost_equal(intersection, [10.0, 20.0, 0.0])
    
    def test_parallel_ray_returns_none(self):
        """평행 광선은 교차점 없음"""
        ray = Ray(
            origin=np.array([0.0, 0.0, 50.0]),
            direction=np.array([1.0, 0.0, 0.0])
        )
        plane = Plane.horizontal_at_height(0.0)
        
        intersection = intersect_ray_plane(ray, plane)
        
        assert intersection is None
    
    def test_backward_ray_returns_none(self):
        """평면 반대 방향으로 향하는 광선"""
        ray = Ray(
            origin=np.array([0.0, 0.0, 50.0]),
            direction=np.array([0.0, 0.0, 1.0])
        )
        plane = Plane.horizontal_at_height(0.0)
        
        intersection = intersect_ray_plane(ray, plane)
        
        assert intersection is None


class TestRotationMatrix:
    
    def test_identity_when_zero_angles(self):
        """모든 각도 0이면 단위 행렬"""
        R = euler_to_rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_rotation_is_orthogonal(self):
        """회전 행렬은 직교"""
        R = euler_to_rotation_matrix(30, 45, 60)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        np.testing.assert_array_almost_equal(np.linalg.det(R), 1.0)


class TestGeoreferencer:
    
    @pytest.fixture
    def nadir_setup(self):
        """Nadir(직하방) 촬영 셋업 - 가장 단순한 케이스"""
        K = np.array([
            [3550.0,    0.0, 2592.0],
            [   0.0, 3550.0, 1944.0],
            [   0.0,    0.0,    1.0]
        ])
        D = np.zeros(5)
        
        intrinsics = CameraIntrinsics(K, D, 5184, 3888)
        
        drone_pos = GeodeticPoint(37.5665, 126.9780, 70.0)
        drone_pose = DronePose(yaw_deg=0, pitch_deg=0, roll_deg=0)
        gimbal_pose = GimbalPose(yaw_deg=0, pitch_deg=-90, roll_deg=0)
        
        input_data = GeoreferencingInput(
            drone_position=drone_pos,
            drone_pose=drone_pose,
            gimbal_pose=gimbal_pose,
            intrinsics=intrinsics,
            ground_height=0.0
        )
        
        return SingleImageGeoreferencer(input_data)
    
    def test_image_center_is_directly_below_drone(self, nadir_setup):
        """Nadir 촬영 시 이미지 중심이 드론 바로 아래"""
        gr = nadir_setup
        
        center_pixel = (5184 // 2, 3888 // 2)
        center_geo = gr.pixel_to_geodetic(center_pixel)
        
        drone_lat = gr.input.drone_position.latitude
        drone_lon = gr.input.drone_position.longitude
        
        assert abs(center_geo.latitude - drone_lat) < 1e-5
        assert abs(center_geo.longitude - drone_lon) < 1e-5
    
    def test_pixel_world_pixel_roundtrip(self, nadir_setup):
        """픽셀 → 월드 → 픽셀 역변환 일치"""
        gr = nadir_setup
        
        test_pixel = (1500.0, 2000.0)
        geo = gr.pixel_to_geodetic(test_pixel)
        recovered_pixel = gr.geodetic_to_pixel(geo)
        
        error = np.sqrt(
            (test_pixel[0] - recovered_pixel[0]) ** 2 +
            (test_pixel[1] - recovered_pixel[1]) ** 2
        )
        assert error < 0.01
    
    def test_gsd_reasonable_value(self, nadir_setup):
        """GSD가 합리적 범위 (수 cm/pixel)"""
        gr = nadir_setup
        gsd = gr.compute_ground_sample_distance()
        
        assert 0.005 < gsd < 0.1
    
    def test_coverage_area_reasonable(self, nadir_setup):
        """커버리지 면적이 합리적 (수 천 m²)"""
        gr = nadir_setup
        coverage = gr.compute_coverage_area()
        
        assert 1000 < coverage < 10000