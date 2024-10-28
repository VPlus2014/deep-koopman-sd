from pathlib import Path
from mathext import *
import cv2

PACK_DIR = Path(__file__).resolve().parent


def main():
    specs = ["LWIR", "MWIR", "SWIR"]
    db_path = PACK_DIR / "MB1"
    # 数据集中的 pitch 是天顶角, 即从NED体轴系 Z-Y 旋转 (-yaw,pitch-90) ，视线与旋转后的 X 轴反向
    roll0_k = 0
    pitch0_k = 90
    yaw0_k = 0
    roll_k = roll0_k
    pitch_k = pitch0_k
    yaw_k = yaw0_k
    dp = 1
    img = None
    show_origin = False
    pix_val_ground = [101] * 3
    bx = 3
    by = 0
    #
    Taff = np.asarray(
        [
            [1, 0, bx],
            [0, 1, by],
        ],
        float,
    )
    while True:
        try:
            key = cv2.waitKey()
            change = True
            # 上下左右
            if key in [82, ord("w"), ord("W")]:  # up
                pitch_k += 1
            elif key in [84, ord("s"), ord("S")]:  # down
                pitch_k -= 1
            elif key in [81, ord("a"), ord("A")]:  # left
                yaw_k -= 1
            elif key in [83, ord("d"), ord("D")]:  # right
                yaw_k += 1
            elif key == ord("q"):  #
                roll_k -= 1
            elif key == ord("e"):  #
                roll_k += 1
            elif key == 32:  # space
                pitch_k = pitch0_k
                yaw_k = yaw0_k
                roll_k = roll0_k
            elif key == 27:  # esc
                break
            else:
                if img is not None:
                    change = False
            if not change:
                continue
            roll_k = roll_k % 360
            pitch_k = pitch_k % 360
            yaw_k = yaw_k % 360

            roll_bl = math.radians(roll_k * dp)
            pitch_bl = math.radians(pitch_k * dp)
            yaw_bl = math.radians(yaw_k * dp)
            r_c, p_c, y_c = rpy_NEDLight2Len(roll_bl, pitch_bl, yaw_bl)

            r_c_deg = int(modin(round(np.rad2deg(r_c)), -180, 360))
            p_c_deg = int(modin(round(np.rad2deg(p_c)), 0, 360))
            y_c_deg = int(modin(round(np.rad2deg(y_c)), 0, 360))

            # rot = rpy2mat(0, pitch_k * dp, yaw_k * dp, deg=True)
            # pitch, yaw = cam_py2mat_inv(rot)

            # pitch_deg = min(max(round(np.rad2deg(pitch)), 0), 359)
            # yaw_deg = min(max(round(np.rad2deg(yaw)), 0), 359)

            print(
                f"(r,p,y)"
                f" body2light:{np.rad2deg(np.ravel((roll_bl, pitch_bl, yaw_bl)))}"
                f" pic:{np.ravel((r_c_deg, p_c_deg, y_c_deg))}"
            )
            pitch_s = f"pitch{p_c_deg}"
            yaw_s = f"yaw{y_c_deg}"
            fname = f"{pitch_s}/{pitch_s}_{yaw_s}.png"
            for spec in specs:
                fn = db_path / spec / fname
                try:
                    img = cv2.imread(str(fn))
                    # pix0 = img[0, 0]
                    # print(f"pix0: {pix0}")
                    if show_origin:
                        cv2.imshow(spec + "_raw", img)
                except Exception as e:
                    print(f"No image found for {fn}")
                    continue
                try:
                    imgH = img.shape[0]
                    imgW = img.shape[1]
                    img = cv2.warpAffine(
                        img, Taff, (imgW, imgH), borderValue=pix_val_ground
                    )
                    Trot = cv2.getRotationMatrix2D(
                        ((imgW - 1) * 0.5, (imgH - 1) * 0.5), r_c_deg, 1
                    )
                    img = cv2.warpAffine(
                        img, Trot, (imgW, imgH), borderValue=pix_val_ground
                    )
                    cv2.imshow(f"{spec}_rot", img)
                except Exception as e:
                    print(f"Error while rotating image: {e}")
        except KeyboardInterrupt:
            break
    pass


if __name__ == "__main__":
    main()
