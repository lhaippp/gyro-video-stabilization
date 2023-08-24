import os
import argparse

import numpy as np

K_cc9_600_800 = np.array([[573.8534, -0.6974, 406.0101], [0, 575.0448, 309.0112], [0, 0, 1]])

# ZOOM X3
# K_cc9_600_800 = 1e3 * np.array([[1.7105, 0.0009, 0.4094],
#                           [0, 1.7099, 0.3189], [0, 0, 0.0010]])
patch = 10
ts = 0.033312
# ts = 0.015138971
# ts /= 1.2


def diffrotation(gyrostamp, gyroidxa, gyroidxb, ta, tb, anglev, gyrogap):
    '''
    R: rotation matrice
    anglev: rate of rotation
    '''
    R = np.eye(3)
    for i in range(gyroidxa, gyroidxb + 1):
        # 计算积分时间dt = tb - ta
        if i == gyroidxa:
            dt = gyrostamp[i] - ta
        elif i == gyroidxb:
            dt = tb - gyrostamp[i - 1]
        else:
            dt = gyrogap[i - 1]

        if gyroidxa == gyroidxb:
            dt = tb - ta
        tempv = dt * (anglev[i - 1, :])  # gyro积分
        theta = np.linalg.norm(tempv)
        tempv = tempv / theta
        tempv = tempv.tolist()
        # gyro和camera坐标轴不同（gyro的x是相机的y）
        skewv = np.array([[0, -tempv[2], tempv[0]], [tempv[2], 0, -tempv[1]], [-tempv[0], tempv[1], 0]])
        nnT = np.array([[tempv[1] * tempv[1], tempv[1] * tempv[0], tempv[1] * tempv[2]],
                        [tempv[1] * tempv[0], tempv[0] * tempv[0], tempv[0] * tempv[2]],
                        [tempv[1] * tempv[2], tempv[0] * tempv[2], tempv[2] * tempv[2]]])
        tempr = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * nnT + np.sin(theta) * skewv
        R = np.dot(R, tempr)
    return R


def imagesHomography(num_frames: int,
                     patch: int,
                     gyrostamp: np.ndarray,
                     gyrogap: np.ndarray,
                     anglev: np.ndarray,
                     framestamp: np.ndarray,
                     K: np.ndarray,
                     delta_T=0):
    Kinv = np.linalg.inv(K)
    hom = np.zeros((patch, 3, 3))
    hom_flat = np.zeros((patch, 9))
    for i in range(patch):
        ta = framestamp[num_frames - 1] + ts * i / patch + delta_T
        tb = framestamp[num_frames] + ts * i / patch + delta_T
        gyroidxa = np.where(gyrostamp > ta)[0][0]
        gyroidxb = np.where(gyrostamp > tb)[0][0]
        gyroidxa = gyroidxa if gyroidxa != 0 else 1
        gyroidxb = gyroidxb if gyroidxb != 0 else 1
        R = diffrotation(gyrostamp, gyroidxa, gyroidxb, ta, tb, anglev, gyrogap)
        hom[i] = K.dot(R).dot(Kinv)
        # normalize homography
        hom[i] /= hom[i][-1, -1]
        hom_flat[i] = hom[i].flatten()
        try:
            assert hom[i, 0, 0] == hom_flat[i, 0], "frame:{} patch:{}:Catch wrong flatten()".format(num_frames, i)
        except Exception as e:
            print(e)
            print(hom_flat[i].shape)
            print("replace invalid homo with np.eyes")
            hom_flat[i] = np.eye(3).flatten()
            continue

        assert hom[i, -1, -1] == 1, "frame:{} patch:{}: not normalize".format(num_frames, i)
    return hom_flat


def make_trainset_source(framestramp_file, gyro_file, K, first_frame, last_frame, delta_T=0):
    framestamp = np.loadtxt(framestramp_file, dtype='float_', delimiter=' ')
    gyro = np.loadtxt(gyro_file, dtype='float_', delimiter=',')

    homs = np.zeros((last_frame - first_frame + 1, patch, 9))

    gyrostamp = gyro[:, -1]
    gyrogap = np.diff(gyrostamp)
    anglev = gyro[:, :3]

    for i in range(len(homs)):
        try:
            hom_flat = imagesHomography(i + first_frame, patch, gyrostamp, gyrogap, anglev, framestamp, K, delta_T)
            homs[i, :, :] = hom_flat
        except AssertionError as e:
            print(e)
            # continue
    return homs


def make_gyro_source_cc9(project_path, filename, idx=[0, 0]):
    GYRO_TRAIN_PATH = os.path.join(project_path, "{filename}".format(filename=filename))
    gyro_frames = [idx]
    gyro_homs = []
    for rotation_frame in gyro_frames:
        print("compute homography between {} and {}".format(rotation_frame[0], rotation_frame[1]))

        gyro_hom = make_trainset_source(
            GYRO_TRAIN_PATH + '/framestamp.txt',
            GYRO_TRAIN_PATH + '/gyro.txt',
            K_cc9_600_800,
            int(rotation_frame[0]),
            int(rotation_frame[1]),
        )

        print(gyro_hom.shape)
        gyro_homs.append(gyro_hom)

    gyro_homs_concat = np.vstack(gyro_homs)
    print(gyro_homs_concat.shape)

    np.save(os.path.join(project_path, "{filename}/gyro_homo_h33_source.npy".format(filename=filename)), gyro_homs_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_path')
    parser.add_argument('--filename')
    parser.add_argument('--idx', nargs='+', required=True)
    args = parser.parse_args()

    make_gyro_source_cc9(args.project_path, args.filename, args.idx)
