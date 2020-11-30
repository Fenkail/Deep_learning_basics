import torch
from matplotlib import pyplot as plt
import numpy as np

epoch = 50000

def f(x,y):
    return (1.5-x+x*y)**2+(2.25-x+x*y*y)**2+(2.625-x+x*y*y*y)**2
def hx(x,y):
    return 2*((1.5-x+x*y)*(y-1) + (2.25-x+x*y*y)*(y*y-1) + (2.265-x+x*y**3)*(y**3-1))
def hy(x,y):
    return 2*((1.5-x+x*y)*x + (2.25-x+x*y*y)*(2*x*y) + (2.265-x+x*y**3)*(3*x*y*y))


def Adam(x, y, ax):
    # 动量与指数衰减的初值
    m_x,m_y = 0,0
    v_x,v_y = 0,0
    # 动量与指数衰减的参数
    b1 = 0.9
    b2 = 0.999
    # 学习率
    n = 0.01
    e = 10e-8
    iters = 0
    X = []
    Y = []
    Z = []
    loss = []
    while iters<epoch:
        iters+=1
        X.append(x)
        Y.append(y)
        Z.append(f(x,y))
        loss.append(f(x,y))
        m_x = b1*m_x + (1-b1)*hx(x,y)
        v_x = b2*v_x + ((1-b2)*hx(x,y))**2
        m_y = b1*m_y + (1-b1)*hy(x,y)
        v_y = b2*v_y + ((1-b2)*hy(x,y))**2
        m_het_x = m_x/(1-b1**iters)
        v_het_x = v_x/(1-b2**iters)
        m_het_y = m_y/(1-b1**iters)
        v_het_y = v_y/(1-b2**iters)
        x = x - n/np.sqrt(v_het_x+e)*m_het_x
        y = y - n/np.sqrt(v_het_y+e)*m_het_y
    print('adam的最终输出：', x, y, f(x,y))
    ax.plot3D(X,Y,Z,'orange',label='adam')
    return loss


def ada_grad(x, y, ax):
    # Adaptive_Gradient
    n = 0.01
    e = 10e-8
    iters = 0
    sum_x_grad = 0
    sum_y_grad = 0
    X = []
    Y = []
    Z = []
    loss = []
    while iters<epoch:
        iters+=1
        X.append(x)
        Y.append(y)
        Z.append(f(x,y))
        loss.append(f(x,y))
        sum_x_grad += (hx(x,y))**2
        sum_y_grad += (hy(x,y))**2
        x = x - n/np.sqrt(sum_x_grad+e)*(hx(x,y))
        y = y - n/np.sqrt(sum_y_grad+e)*(hy(x,y))
    print('ada_grad的最终输出：',x, y,f(x,y))
    ax.plot3D(X,Y,Z,'black', label='ada_grad')
    return loss

def momentum(x, y, ax):
    # 学习率
    n = 0.01
    # 前一轮动量的缩放
    a = 0.9
    # 初始动量
    vx = 0
    vy = 0
    iters = 0
    X = []
    Y = []
    Z = []
    loss = []
    while iters<epoch:
        iters+=1
        X.append(x)
        Y.append(y)
        Z.append(f(x,y))
        loss.append(f(x,y))
        vx = a*vx - n*hx(x,y)
        x = x + vx
        vy = a*vy - n*hy(x,y)
        y = y + vy
    print('momentum的最终输出：',x,y,f(x,y))
    ax.plot3D(X,Y,Z,'gray',label='momentum')
    return loss




def rmsprop(x, y, ax):
    n = 0.01
    e = 10e-8
    iters = 0
    sum_x_grad = 0
    sum_y_grad = 0
    X = []
    Y = []
    Z = []
    a = 0.1
    loss = []
    while iters<epoch:
        iters+=1
        X.append(x)
        Y.append(y)
        Z.append(f(x,y))
        loss.append(f(x,y))
        sum_x_grad = a*sum_x_grad + (1-a)*(hx(x,y))**2
        sum_y_grad = a*sum_y_grad + (1-a)*(hy(x,y))**2
        x = x - n/np.sqrt(sum_x_grad+e)*(hx(x,y))
        y = y - n/np.sqrt(sum_y_grad+e)*(hy(x,y))
    print('rmsprop的最终输出：',x, y,f(x,y))
    ax.plot3D(X,Y,Z,'blue', label='rmsprop')
    return loss

if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = np.arange(-3, 3, 0.025)
    y = np.arange(-3, 3, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.square(1.5-X+X*Y)+np.square(2.25-X+X*Y**2)+np.square(2.625-X+Y*Y*3)
    ax.plot_surface(X,Y,Z,cmap='rainbow',alpha=0.3)
    plt.title("optim analysis")
    # 初值
    x = 2
    y = -1
    loss_m = momentum(x,y,ax)
    loss_ag = ada_grad(x,y,ax)
    loss_am = Adam(x,y,ax)
    loss_rm = rmsprop(x, y, ax)
    plt.legend(loc='upper left')
    plt.savefig('./optim/test.jpg')
    

    x = np.linspace(0,epoch,num=epoch)
    fig,ax = plt.subplots()
    ax.plot(x,loss_m,label='momentum')
    ax.plot(x,loss_am,label='adam')
    ax.plot(x,loss_ag,label='ada_grad')
    ax.plot(x,loss_rm,label='rmsprop')
    ax.set_title('optim speed')
    ax.legend()
    plt.savefig('./optim/speed.jpg')
