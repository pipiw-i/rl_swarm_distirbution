# -*- coding: utf-8 -*-
# @Time : 2022/5/26 下午8:06
# @Author :  wangshulei
# @FileName: boid.py
# @Software: PyCharm
# -*- coding:utf-8 -*-
import sys, argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist  # 计算点之间的距离
from numpy.linalg import norm
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

width, height = 640, 480  # 设置屏幕上模拟窗口的宽度和高度


class Boids:
    def __init__(self, N):
        """ initialize the Boid simulation"""
        # 初始化位置和速度
        self.pos = [width / 2.0, height / 2.0] + 10 * np.random.rand(2 * N).reshape(N,2)
        # '''创建一个 numpy 数组 pos，对窗口中心加上 10 个单位以内的随机偏移。
        # 代码np.random.rand（2 * N）创建了一个一维数组，包含范围在[0，1]的 2N 个随机数。
        # 然后 reshape()调用将它转换成二维数组的形状（N，2），它将用于保存类鸟群个体的位置。'''
        # normalized random velocities
        angles = 2 * math.pi * np.random.rand(N)
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles))))
        self.N = N  # 生成一个数组，包含 N 个随机角度，范围在[0, 2pi]，
        # min dist of approach
        self.minDist = 25.0
        # max magnitude of velocities calculated by "rules"
        self.maxRuleVel = 0.03
        # max maginitude of final velocity
        self.maxVel = 2.0

    def tick(self, frameNum, pts, beak):
        """Update the simulation by one time step."""
        # 得到目前所有个体的相对距离
        self.distMatrix = squareform(pdist(self.pos))  # 用 squareform()和 pdist()方法来计算一组点之间两两的距离
        # apply rules:
        self.vel += self.applyRules()
        self.limit(self.vel, self.maxVel)
        self.pos += self.vel
        self.applyBC()
        # update data
        pts.set_data(self.pos.reshape(2 * self.N)[::2],
                     self.pos.reshape(2 * self.N)[1::2])
        vec = self.pos + 10 * self.vel / self.maxVel
        beak.set_data(vec.reshape(2 * self.N)[::2],
                      vec.reshape(2 * self.N)[1::2])

    def limitVec(self, vec, maxVal):
        """limit magnitide of 2D vector"""
        mag = norm(vec)
        if mag > maxVal:
            vec[0], vec[1] = vec[0] * maxVal / mag, vec[1] * maxVal / mag

    def limit(self, X, maxVal):
        """limit magnitide of 2D vectors in array X to maxValue"""
        for vec in X:
            self.limitVec(vec, maxVal)

    def applyBC(self):
        """apply boundary conditions"""
        deltaR = 2.0  # 该行中的deltaR提供了一个微小的缓冲区，它允许类鸟群个体开始从相反方向回来之前移出小块之外一点，从而产生更好的视觉效果
        for index, coord in enumerate(self.pos):
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR
            if ((coord[1] - 225) ** 2 + (coord[0] - 225) ** 2) <= 3600:
                self.vel[index] = -self.vel[index] * 2

    def applyRules(self):
        # apply rule #1 - Separation 分离
        D = self.distMatrix < 25.0  # 判断得到的距离是否小于25，这里d是一个bool阵列，小于25为True
        # self.pos 所有个体的位置，dot，矩阵乘法
        x = D.dot(self.pos)
        y = D.sum(axis=1)
        z = self.pos * D.sum(axis=1).reshape(self.N, 1)
        vel = self.pos * D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)

        # different distance threshold
        D = self.distMatrix < 50.0

        # apply rule #2 - 列队
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2

        # apply rule #1 - 聚集
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3

        return vel

    def buttonPress(self, event):
        """event handler for matplotlib button presses"""
        # left click - add a boid
        if event.button is 1:
            self.pos = np.concatenate((self.pos,
                                       np.array([[event.xdata, event.ydata]])),
                                      axis=0)
            # random velocity
            angles = 2 * math.pi * np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis=0)  # 拼接
            self.N += 1
            # right click - scatter
        elif event.button is 3:
            # add scattering velocity
            self.vel += 0.1 * (self.pos - np.array([[event.xdata, event.ydata]]))


def tick(frameNum, pts, beak, boids):
    # print frameNum
    """update function for animation"""
    boids.tick(frameNum, pts, beak)
    return pts, beak


# main() function
def main():
    # use sys.argv if needed
    print('starting boids...')

    parser = argparse.ArgumentParser(description="Implementing Craig Reynold's Boids...")
    # add arguments
    parser.add_argument('--num-boids', dest='N', required=False)
    args = parser.parse_args()

    # number of boids
    N = 100
    if args.N:
        N = int(args.N)

    # create boids
    boids = Boids(N)

    # setup plot
    fig = plt.figure(facecolor='pink')
    ax = plt.axes(xlim=(0, width), ylim=(0, height), facecolor='lightskyblue')
    ax.add_patch(patches.Rectangle((200, 200), 50, 50, linewidth=1, edgecolor='b', facecolor='b'))
    pts, = ax.plot([], [], markersize=10,
                   c='k', marker='o', ls='None')
    beak, = ax.plot([], [], markersize=4,
                    c='r', marker='o', ls='None')

    anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, boids),
                                   interval=50)

    # add a "button press" event handler
    cid = fig.canvas.mpl_connect('button_press_event', boids.buttonPress)

    plt.show()


# call main
if __name__ == '__main__':
    main()