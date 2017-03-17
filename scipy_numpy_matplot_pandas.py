import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#python



# x=pd.Series(np.random.randn(5),index=['a','b','c','d','e'])
# print (x)
# print(np.exp(x))
# message='hello,world,how,are,you'
# s=message.split(',')
# print(s)
#
# import sys
# a=sys.argv
# print(a)
#
# import demo
# demo.print_a()
# help(demo)
#
# import sys
# print(sys.path)


#文件读取
# f=open('demo.py','r')
# print(type(f))
# for line in f:
#     print(line)

import os#系统方面的操作
print(os.getcwd())

print(os.listdir(os.curdir))
print(os.path.abspath('README.md'))
a='README.md'
print(os.path.dirname(a))
print(os.path.basename(a))
print(os.path.split(a))
print(os.path.splitext(os.path.basename(a)))
print(os.path.exists(a))
print(os.path)
print(os.path.expanduser('~'))
print(os.path.join(os.path.expanduser('~'),'local','bin'))
os.system('ls')

import sys#系统中和python有关的信息
print(sys.platform)
print(sys.version)
print(sys.prefix)
print(sys.getdefaultencoding())
# while True:
#     try:
#        x=int(input('please enter a number:'))
#        break
#     except ValueError as e:
#         print('that was not valid numver .Try again...')
#

#class

class Student(object):
    def __init__(self,name):
        self.name=name
    def set_age(self,age):
        self.age=age
    def set_major(self,major):
        self.major=major

ann=Student('yangchao')
ann.set_age(14)
ann.set_major('computer')
print(ann.name)







#numpy


import numpy as np

a=np.array([0,1,2,3])
print(a)
print([i**2 for i in a])
print(a**2)

a=np.arange(100)
print(a**2)
print(a.shape)
print(a.ndim)
b=np.array([[1,2,3],[3,4,5]])#两行三列
print(b)
print(b.shape)
print(len(b))#是第一个维度的长度
c=np.array([[[2,3,3,7],[3,4,7,7],[2,3,4,3]],[[2,3,4,5],[4,5,3,4],[1,6,8,6]]])
print(c.shape)#输出2,3,4
print(len(c))#第一个维是三维的第三个维。
print(c.ndim)
print(c)
a=np.linspace(0,1,3)
print(a)
d=np.linspace(0,1,5,endpoint=False)
print(d)
a=np.ones((3,3))
b=np.zeros((2,2))
print(b)
c=np.eye(3)#这个确定是单位矩阵，只用说是多大的就行
print(c)
d=np.diag(np.array([1,2,3]))#这个确定是对角矩阵，只需要指定对角的值就行
print(d)


a=np.random.rand(4)
b=np.random.randn(4)
print(a)
print(b)
a=np.array(np.random.randn(3))
print(a)
np.random.seed(100)
b=np.random.randn(4)
print(b)
a=np.array([1,2,3])
#data type
print(a.dtype)

x=np.linspace(0,3,20)
y=np.linspace(0,9,20)
# import matplotlib.pyplot as pyplot
# plt.plot(x,y)
#
# plt.plot(x,y,'o')
# plt.show()
#
# image=np.random.randn(30,30)
# plt.imshow(image,cmap=plt.cm.hot)
# plt.colorbar()
# plt.show()

np.random.seed(3)
a=np.random.random_integers(0,29,15)
print(a)
b=(a%3==0)
print(b)
c=b[::2]
print(c)
mask=(a%3==0)
print('this is mask:',mask)
a[a%3==0]=-1#它能够根据true项来进行索引
print(a)
a=np.arange(10)+15
b=np.array([[3,5,4],[3,6,8]])

print(a[b])#生成的矩阵是取b的shap,a的值的。

print(np.log(a))
print(np.exp(a))
a=np.random.random_integers(0,100,6)
b=np.random.random_integers(0,100,6)
print(a)
print(b)
print(a==b)
print(a>b)
print(np.array_equal(a,b))
d=np.triu([[1,2,3],[3,2,3],[1,1,2]])#一个矩阵，只取它的上三角部分。
print(d)
c=np.ones((3,3))
print(c)

print(d.T)#转置,由于缓存的原因，只能处理小数据。

print(np.allclose(a,b,10,0))


x=np.array([2,3,4,2])
y=np.array([[[1,3],[-1,4]],[[5,2],[3,4]]])
print('sum:',np.sum(x))
print(np.sum(y,axis=0))
print(np.min(x))
print(np.argmin(y))#输出的是最小的位置，索引
print(np.mean(x))
print(np.median(x))#中位数
print(np.std(x))#标准差
print(np.sqrt(np.sum((x-2.75)**2)/4))

print(np.cumsum(x))#从头到尾的累加。每一步的和都显示了。是一个tensor

# x=np.random.randn(10,10)
# plt.imshow(x)
# plt.colorbar()
# plt.show()

#walker

# n_stories=1000
# t_max=200
# t=np.arange(t_max)
# steps=2*np.random.random_integers(0,1,(n_stories,t_max))-1
# print(steps)
# b=np.unique(steps)
# print(b)
# position=np.cumsum(steps,axis=1)
# print(position)
# sq_distance=position**2
# mean_position=np.mean(position,axis=0)
# mean_sq_position=np.mean(sq_distance,axis=0)
# print(mean_sq_position)
# print('this is ')
# print(mean_position)
# print('this is')
# print(np.sqrt(mean_sq_position))
# plt.figure(figsize=(4,3))
#
# plt.plot(t,mean_position)
# plt.show()
# plt.plot(t,np.sqrt(mean_sq_position),'g.',t,np.sqrt(t),'y-')
# plt.show()
# y=np.random.randn(4,4)
# t=np.arange(4)
# plt.figure()
# plt.plot(t,y)
# plt.show()


x=np.random.randint(0,10,(1,5))
y=np.random.randint(0,10,(5,1))
print('x',x)
print('y',y)
print('x+y',x+y)#这就说明


x=np.arange(0,40,10)#取不到末尾
y=np.arange(10)
print(y)
print(x)
print(x.shape)#行向量,一维的，如果要是列向量，那一定是二维的。
b=x[:,np.newaxis]
print(b.shape)
#
# x,y=np.arange(5),np.arange(5)[:,np.newaxis]
# distance=np.sqrt(x**2+y**2)
# plt.figure()
# # plt.imshow(distance)
# plt.pcolor(distance)#这个会把图片反转过来
# plt.colorbar()
# plt.show()

# x,y=np.ogrid[0:5,0:5]
# print(x,y)
# print(x.shape,y.shape)
# distance=np.sqrt(x**2+y**2)
# plt.pcolor(distance)
# plt.colorbar()
# plt.show()


a=np.random.randint(0,10,(2,3))
print('a',a)
print('a.T',a.T)
print('ravel a',np.ravel(a))
print('ravel a.t',np.ravel(a.T))
print('ravel a order=f',np.ravel(a,order='F'))#flattening 压平操作

#inverse operation to flattening

print(np.reshape(a,(3,2)))#其实都是压平后再重新构造。按照行

b=np.reshape(a,(3,2))
b[0,1]=50
print(b)
print(a)
c=np.transpose(a)
d=np.reshape(c,(2,3))
d[1,1]=40
print('c',c)
print(d)
print(a)
c[0,0]=30
print(c)
print(a)


print('\n\n\n new')

a=np.random.randint(10,100,4*3*2)
print(a)
print(a.reshape(4,3,2))
c=np.reshape(a,(4,3,2))
b=np.arange(1,40,4)#1到40之间间隔4取所有的值
print(b)

print(c)
print(c.flatten())
print(np.size(c))#size就是总长度。而shape是形状的意思。


print('\n\n')


a=np.random.randint(10,100,(4,3))
c=np.ravel(a)#这是一个view
c[0]=-1
d=a.flatten()#这是一个复制
print(d)
d[1]=-2
print(a)
# print(a)
# print(c)
# print(a)

# x=np.ma.array([1,2,3,4],mask=[0,1,0,1])
# print(x)#mask 通过面具来过滤一些数据
#
# p=np.poly1d([3,2,-1])
# print(p)
# print(p(1))#还能对多项式求值，这时候p相当于一个函数
# print(np.roots(p))
# print(p.order)
#
# x=np.linspace(0,1,20)
# y=np.cos(x)+3*np.random.rand(20)
# d=np.polyfit(x,y,4)
# print(d)
# p=np.poly1d(np.polyfit(x,y,4))
# t=np.linspace(0,1,200)
# plt.figure()
# plt.plot(x,y,'ro',t,p(t),'g-')
# plt.show()

#多项式类，这个借口感觉比poly1d 好


# print('this is polynomial\n')
# p=np.polynomial.Polynomial([-1,2,3])
# print(p(0))
# print(p)
# print(p.roots())
# print(p.degree())
#
# x=np.linspace(-1,1,2000)
# y=np.cos(x)+0.3*np.random.rand(2000)
# p=np.polynomial.Chebyshev.fit(x,y,90)
# t=np.linspace(-1,1,2000)
#
# plt.plot(x,y,'r.')
# plt.plot(t,p(t),'k-',lw=3)
# plt.show()


#loadtxt
# x=np.random.randn(3,4)
# print(x)
# np.savetxt('data/x.txt',x)
# y=np.loadtxt('data/x.txt')
# print(y)
# plt.imshow(y)
# plt.colorbar()
# plt.show()


# img=plt.imread('data/me.png')
# print(img.shape,img.dtype)

# plt.imshow(img)
# plt.colorbar()
# plt.show()
# plt.savefig('data/plot_me.png')#这个保存是有问题的
# plt.imsave('data/im_me',img[:,:,0],cmap=plt.cm.gray)

# img=plt.imread('data/plot_me.png')
# plt.imshow(img)
# plt.show()

# img=plt.imread('data/im_me')
# plt.imshow(img)
# plt.show()
# print(img.shape)


# img=plt.imread('data/me.png')
# plt.imsave('data/1_me.png',img[:,:,1],cmap=plt.cm.gray)
#
# img1=plt.imread('data/3_me.png')
# plt.imshow(img1)
# plt.show()



#exercise
x=np.arange(1,16)
y=np.reshape(x,(5,3),order='F')
print(y)
z=y[1::2,:]
print(z)

a=np.arange(25).reshape(5,5)
print(a)

np.argsort
b=np.array([1,5,10,15,20])
b=b[:,np.newaxis]
print(b)
z=np.arange(5)
print(z)
a=b+z
print(a)

a=np.random.random_sample(size=(10,3))
print(a)

b=np.abs(a-0.5)
print(b)
d=np.argsort(b,axis=0)
print(d)
c=d[0,:]
e=a[c,np.arange(3)]
print(e)

from scipy import misc
face=misc.face(gray=True)





#matplotlib


# plt.imshow(face,cmap=plt.cm.gray)
# plt.show()

# print(face.shape)
# check=face[100:-100,100:-100]
# plt.imshow(check,cmap=plt.cm.gray)
# plt.show()

# x,y=face.shape
# print(x,y)
# sr,sd=np.ogrid[:x,:y]
# mask=(sr-300)**2+(sd-660)**2>230**2
# face[mask]=0
# plt.imshow(face,cmap=plt.cm.gray)
# plt.show()

#
# x=np.linspace(-4,4,200)
# y=np.cos(x)
#
# plt.plot(x,y,color='blue',linewidth=2.0,linestyle='-')
# plt.xlim(-4,4)
# plt.xticks(np.linspace(-4,4,9))
# plt.ylim(-1,1)
# plt.grid()
# ax=plt.gca()
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('right')
# plt.yticks(np.linspace(-1,1,3,endpoint=True))
# plt.legend('this')
# plt.fill_between(x,y)
# plt.show()



#mandelbrot

# N_max=50
# some_threshold=50
# x=np.linspace(-2,1,200)
# y=np.linspace(-1.5,1.5,200)
# y=y[:,np.newaxis]
# c=x+1j*y
# # print(c)
# z=0
# for i in np.arange(N_max):
#     z=z**2+c
# print(z)
# d=np.abs(z)
# print(d)
# mask=np.abs(z)<some_threshold
# print(mask)
#
# plt.imshow(mask,extent=[-2,1,-1.5,1.5])
#
# plt.show()
#
#
# plt.imsave('data/mandelbrot.png',mask)
#
# n=1024
# x=np.random.uniform(2,3,n)
# y=np.random.uniform(3,4,n)
# z=np.random.normal(0,1,n)
# d=np.random.normal(0,1,n)
# t=np.arctan2(z,d)
# plt.scatter(z,d,s=100,c=t,alpha=0.5)
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
# plt.colorbar()
#
# # plt.scatter(x,y)
# plt.show()


# x=np.arange(10)
# y=(1-x/10)*np.random.rand(10)
#
# y1=(1-x/10)*np.random.rand(10)
#
# plt.bar(x,+y,facecolor='#9999ff',edgecolor='white')
# plt.bar(x,-y,facecolor='#ff9999',edgecolor='white')
# plt.show()


# def f(x,y):
#     return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
#
# n=1024
# x=np.linspace(-3,3,n)
# y=np.linspace(-3,3,n)
# X,Y=np.meshgrid(x,y)
# print(X,Y)
#
# plt.contourf(X,Y,f(X,Y),alpha=0.7)
# c=plt.contour(X,Y,f(X,Y))
# plt.clabel(c)
# plt.show()




#scipy
import scipy
from scipy import io as scio

# x=scio.loadmat()
# scio.savemat()
# np.recfromcsv()


import scipy.linalg as linalg

a=np.array([[1,2],[3,4]])
b=linalg.det(a)
d=linalg.inv(a)
print(b)
print(d)


e=np.dot(a,d)
print(e)

arr=np.arange(9).reshape((3,3))+np.diag([1,0,1])
s,k,g=linalg.svd(arr)
print(s,k,g)

dd=s.dot(np.diag(k)).dot(g)
print(dd)
print(arr)



#fftpack

# import scipy.fftpack as fftpage
#
# time_step=0.02
# period=5
# time_vec=np.arange(0,20,time_step)
# sig=np.sin(2*np.pi/period*time_vec)+\
#     0.5*np.random.randn(time_vec.size)
# plt.figure(2)
# plt.subplot(2,2,1)
# plt.plot(time_vec,sig)
# plt.show()
#
#
# sample_freq=fftpage.fftfreq(sig.size,d=time_step)
#
# sig_fft=fftpage.fft(sig)
#

#
# pidxs=np.where(sample_freq>0)
# freqs=sample_freq[pidxs]
# power=np.abs(sig_fft)[pidxs]
# freq=freqs[power.argmax()]
# zz=np.arange(sample_freq.size)
# sig_fft[np.abs(sample_freq)>freq]=0
# main_sig=fftpage.ifft(sig_fft)
# plt.subplot(2,2,2)
# plt.plot(zz,sample_freq)
# plt.subplot(2,2,3)
# plt.plot(time_vec,main_sig)
# plt.subplot(2,2,4)
# dd=np.arange(freqs.size)
# plt.plot(dd,freqs)
#
# plt.show()



import scipy.optimize as op

#optimize


# def f(x):
#     return x**2+10*np.sin(x)
#
# x=np.arange(-10,10,0.1)
# plt.plot(x,f(x))
# plt.show()
#
# op.fmin_bfgs(f,3)
# print(r'let"s print the bashinhopping')
# print(op.basinhopping(f,0))


def f2(x,a,b):
    return a*x**2+b*np.sin(x)

def f(x):
    return x**2+10*np.sin(x)

# x=np.linspace(-10,10,200)
# y=f(x)+np.random.randn(200)
# params,d=op.curve_fit(f2,x,y,[2,2])
# plt.figure(3)
# plt.subplot(2,2,1)
# plt.plot(x,f(x))
# plt.subplot(2,2,2)
# plt.plot(x,y)
# plt.subplot(2,2,3)
# plt.plot(x,f2(x,params[0],params[1]))
# plt.show()


#import matplotlib.pylab as pl


#exercise :curve fitting of temperature data
# x_max=np.array([17,19,21,28,33,38,37,37,31,23,19,18])
# x_min=np.array([-62,-59,-56,-46,-32,-18,-9,-13,-25,-46,-52,-58])
# t=np.arange(1,13,1)
# plt.plot(t,x_max,'r-',t,x_min,'g.')
# # plt.show()
#
# def f_fit(x,a,b,c,d):
#     return a*np.cos(b*x+c)+d
#
# proms,dd=op.curve_fit(f_fit,t,x_max)
# print(proms)
# plt.plot(t,f_fit(t,proms[0],proms[1],proms[2],proms[3]),'y-',linewidth=1)
# proms1,dd1=op.curve_fit(f_fit,t,x_min)
# plt.plot(t,f_fit(t,proms1[0],proms1[1],proms1[2],proms1[3]),'b-')
# plt.show()
#


def f(x,y):
    return (4-2.1*x**2+x**4/3)*x**2+x*y+(4*y**2-4)*y**2

# x=np.linspace(-2,2,200)
# y=np.linspace(-1,1,200)
# X,Y=np.meshgrid(x,y)

# from mpl_toolkits.mplot3d import Axes3D
#
# fig=plt.figure()
# ax=Axes3D(fig)
# ax.plot_surface(X,Y,f(X,Y),rstride=1,cstride=1,cmap=plt.cm.hot)
# ax.contourf(X,Y,f(X,Y),zdir='z',offset=-1)
#
# plt.show()

np.histogram


import scipy.stats as stats



a=np.random.normal(size=1000)
# b=np.random.rand(10)
# print(b[1:])
# print('\n')
# print(b[-3:-1])
# print('\n')
# print(b)
#
#
#
# bins=np.arange(-4,5)
# histogram=np.histogram(a,bins,normed=True)[0]#当normed或者density就返回概率密度函数值
# print(histogram)
# bins=0.5*(bins[1:]+bins[:-1])
#
#
# b=stats.norm.pdf(bins)#标准正太分布
# plt.plot(bins,histogram)
# plt.plot(bins,b)
# plt.show()


mu,sigma=stats.norm.fit(a)#如果知道分布类型，就能狗根据数据所在分布的数据算出来它的方差和期望
print(mu,sigma)

aa=np.random.gamma(shape=1,size=1000)
bins=np.arange(-4,5)
histogram=np.histogram(aa,bins,normed=True)
print(histogram)
print('\n')
print(histogram[0])
bins=0.5*(bins[1:]+bins[:-1])
b=stats.gamma.pdf(bins,a=1)
# plt.plot(bins,histogram[0])
# plt.plot(bins,b)

shape=stats.gamma.fit(aa)
print(shape)
# plt.show()

print('\n')

print(np.median(aa),'\n')
print(np.median(a))

print(stats.scoreatpercentile(a,50),'\n')



a=np.random.rand(1000)
b=np.random.normal(1,2,100)

print(stats.ttest_ind(a,b))


print('\n')




from scipy.interpolate import interp1d




measure_test=np.linspace(0,1,10)

noise=(np.random.random(10)*2-1)*1e-1
mesu=np.sin(2*np.pi*measure_test)+noise


linear_interp=interp1d(measure_test,mesu)
cuomputed_tim=np.linspace(0,1,50)

linear_result=linear_interp(cuomputed_tim)
cubic_interp=interp1d(measure_test,mesu,kind='cubic')
cubic_result=cubic_interp(cuomputed_tim)
print(linear_result)
print('\n')
measure=np.linspace(0,1,50)
plt.plot(measure_test,mesu,'ro')

plt.plot(measure,linear_result)
plt.plot(measure,cubic_result)


plt.show()
