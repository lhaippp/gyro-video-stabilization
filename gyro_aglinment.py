import cv2
import os
import argparse
import imageio
import math

import numpy as np

patch = 10


def psnr(img1, img2):
    mse = cv2.norm(img1, img2, cv2.NORM_L2SQR)
    mse = mse / (img1.shape[0] * img1.shape[1])
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def transformImage(img, hom, patch):
    height, width = img.shape[:2]
    num = height // patch
    out_img = np.zeros((height, width, 3))
    for row in range(hom.shape[0]):
        temp = cv2.warpPerspective(img, hom[row, :, :], (width, height))
        if row == patch - 1:
            out_img[row * num:] = temp[row * num:]
            continue
        out_img[row * num:row * num + num] = temp[row * num:row * num + num]
    return out_img


def warp_img_make_gif(frames_path_ois_off, match_frames_path_ois_off, gyro_homos, gif_path=None, thres=None):
    img_cc9pro = (cv2.imread(i) for i in frames_path_ois_off)
    match_img_cc9pro = (cv2.imread(i) for i in match_frames_path_ois_off)

    total_psnr_orig = 0
    total_psnr_gyro = 0
    valid_frames = []
    cnt = 0

    for i, _ in enumerate(gyro_homos):
        # if i not in [734, 1404, 1808, 2037]:
        #     continue
        make_gif = True
        img_orig = next(img_cc9pro)
        img_match = next(match_img_cc9pro)

        warp_gyro = transformImage(img_orig, gyro_homos[i].reshape(patch, 3, 3), patch)

        _psnr_orig = psnr(np.float32(img_orig[15:-15, 15:-15]), np.float32(img_match[15:-15, 15:-15]))
        _psnr_gyro = psnr(np.float32(warp_gyro[15:-15, 15:-15]), np.float32(img_match[15:-15, 15:-15]))
        print("{:.2f}-{:.2f}".format(_psnr_orig, _psnr_gyro))

        total_psnr_orig += _psnr_orig
        total_psnr_gyro += _psnr_gyro

        # sort by gap between original PSNR and gyro_warp PSNR
        _psnr_gap = _psnr_gyro - _psnr_orig

        cnt += 1

        if make_gif:
            with imageio.get_writer(os.path.join(gif_path, 'frame-{}-{}-{:.2f}.gif'.format(i, i + 1, _psnr_gap)), mode='I',
                                    duration=0.5) as writer:
                x = np.concatenate((cv2.copyMakeBorder(img_match[15:-15, 15:-15], 10, 10, 10, 10, cv2.BORDER_CONSTANT),
                                    cv2.copyMakeBorder(img_match[15:-15, 15:-15], 10, 10, 10, 10, cv2.BORDER_CONSTANT)),
                                   axis=1)
                y = np.concatenate((cv2.copyMakeBorder(img_orig[15:-15, 15:-15], 10, 10, 10, 10, cv2.BORDER_CONSTANT),
                                    cv2.copyMakeBorder(warp_gyro[15:-15, 15:-15], 10, 10, 10, 10, cv2.BORDER_CONSTANT)),
                                   axis=1)

                writer.append_data(x[..., ::-1].astype(np.uint8))
                writer.append_data(y[..., ::-1].astype(np.uint8))

    print(total_psnr_orig / cnt)
    print(total_psnr_gyro / cnt)
    return valid_frames



def image_alignment_with_gyro(data_path, idx, gif_path, split="RE"):
    gyro_homos = np.load(os.path.join(data_path, "gyro_homo_h33_source.npy"))
    _frame_path = os.path.join(data_path, "reshape")
    frames_path_ois_off = [os.path.join(_frame_path, "{}_frame-{}.jpg".format(split, i)) for i in range(int(idx[0]), int(idx[1]) + 1)]

    match_frames_path_ois_off = [
        os.path.join(_frame_path, "{}_frame-{}.jpg".format(split, i + 1)) for i in range(int(idx[0]),
                                                                                         int(idx[1]) + 1)
    ]

    valid_frames = warp_img_make_gif(frames_path_ois_off, match_frames_path_ois_off, gyro_homos, gif_path=gif_path)
    # print(valid_frames)
    # np.save("Rain_streak_effect_valid_frames", valid_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--idx', nargs='+', required=True)
    parser.add_argument('--split')
    parser.add_argument('--gif_path', default=None)
    args = parser.parse_args()

    image_alignment_with_gyro(args.data_path, args.idx, args.gif_path, args.split)
