import numpy as np
import matplotlib.pyplot as plt
import py_camera_model as cam


def verify_brown_model():
    print("=== Testing Brown Camera Model ===")

    # 1. 初始化模型 (模拟一个广角畸变相机)
    model = cam.BrownCameraModel()
    width, height = 1920, 1080

    K = np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    # 典型的桶形畸变 (Barrel Distortion) 参数
    # k1, k2, p1, p2, k3
    D = np.array([-0.3, 0.1, 0.001, 0.001, 0.0], dtype=np.float32)

    model.init(width, height, K, D)
    print(f"Model Name: {model.name()}")

    # 2. 生成网格点 (用于可视化畸变)
    # 在 3D 空间 Z=1 平面上生成规则网格
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1.0, 1.0, 15))
    rays = np.stack(
        [grid_x.flatten(), grid_y.flatten(), np.ones_like(grid_x.flatten())], axis=1
    )

    # 3. 投影 (Ray -> Pixel) 并收集数据
    pixels_distorted = []
    valid_rays = []

    for ray in rays:
        success, px = model.ray_to_pixel(ray)
        if success and 0 <= px[0] < width and 0 <= px[1] < height:
            pixels_distorted.append(px)
            valid_rays.append(ray)

    pixels_distorted = np.array(pixels_distorted)
    valid_rays = np.array(valid_rays)

    # 4. 可视化：绘制畸变网格
    plt.figure(figsize=(10, 6))
    plt.scatter(
        pixels_distorted[:, 0],
        pixels_distorted[:, 1],
        s=5,
        c="r",
        label="Distorted Grid",
    )
    plt.xlim(0, width)
    plt.ylim(height, 0)  # 图像坐标系 Y 轴向下
    plt.title(f"Brown Distortion Visualization\nGrid Lines should look 'curved'")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("brown_distortion_check.png")
    print("Saved visualization to brown_distortion_check.png")

    # 5. Round-Trip 精度验证 (Pixel -> Ray -> Pixel)
    # 验证数学反解（不动点迭代）的精度
    print("\n--- Running Round-Trip Accuracy Test ---")
    max_error = 0.0

    # 选取图像中心的点进行测试
    test_px = np.array([960.0, 540.0], dtype=np.float32)
    # 选取一个边缘点 (畸变最大处)
    test_px_edge = np.array([100.0, 100.0], dtype=np.float32)

    for px_in in [test_px, test_px_edge]:
        # Pixel -> Ray
        ok1, ray_out = model.pixel_to_ray(px_in)
        if not ok1:
            continue

        # Ray -> Pixel (Reprojection)
        ok2, px_reproj = model.ray_to_pixel(ray_out)

        err = np.linalg.norm(px_in - px_reproj)
        max_error = max(max_error, err)

        print(
            f"Pixel: {px_in} -> Ray: {ray_out[:2]}... -> Reproj: {px_reproj} | Err: {err:.6f} px"
        )

    if max_error < 0.1:
        print(
            f"\n[PASSED] Round-trip error {max_error:.6f} is within sub-pixel tolerance."
        )
    else:
        print(
            f"\n[FAILED] Round-trip error {max_error:.6f} is too high! Check iteration logic."
        )


if __name__ == "__main__":
    verify_brown_model()
