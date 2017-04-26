# this is mylearning of tensorflow scipy,numpy,matplotlib
我用scipy_lecture上的教程。其实上面还有python 的教程，已经做得很棒了。
## cmd+?可以多行注释
## numpy数组的二元运算，可以不是相同维度的，他们是通过扩展到相同维度进行的

## 原来numpy里面还有多项式操作，这仿造matlab简直一流啊


\[
f(x)=\beta x
\]

# scipy

## scipy.io
<<<<<<< HEAD
<<<<<<< HEAD

    io.loadmat/io.savemat
    from scipy import misc
    msic.imread()


## scipy.linalg :linear algebra operations

## 切片
切片操作是能取到到开头，但是取不到结尾。比如b[1:]能取到b[1]，但是b[:-1]是取不到最后一个的。

[这是一个链接](http://baidu.com)
# tensorflow 的安装
**必须要安装java**

    brew cask install java

**必须安装bazel**它是google的编译器

    brew install bazel

**必须安装swig**

  brew install swig

**执行安装**

    $Git clone --recurse-submodules https://github.com/tensorflow/tensorflow
    $cd tensorflow
    #必须先配置
    $./configure
    #编译
    $bazel build --copt=-march=native -c opt //tensorflow/tools/pip_package:build_pip_package
    %打包成pip 安装包whl
    $bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    %安装
    pip install /tmp/tensorflow_pkg/tensorflow-1.1.0rc1-cp35-cp35m-macosx_10_6_x86_64.whl
    %这个生成的文件是编译后生成的，中间过程可能会出现一些警告，不用担心。

在配置的时候一定要主义安装目录和使用的python解释器。它会根据自己电脑中的硬件，和软件生成合适的包
=======
=======
>>>>>>> origin/first
io.loadmat/io.savemat
from scipy import misc
msic.imread()

## scipy.linalg :linear algebra operations

## 切片操作是能取到到开头，但是取不到结尾。比如b[1:]能取到b[1]，但是b[:-1]是取不到最后一个的。
<<<<<<< HEAD
>>>>>>> origin/first
=======
>>>>>>> origin/first

# tensorflow 继续学习
## 首先，我将tensorflow 书本中的mnist 写了一遍。
老生常谈，没什么好说的。简单。为啥简单呢，没有背景，没有多余的因素。字丑了点，但是你毕竟单一。
## 然后，学习了自编码，
  * 这里面问题可多了。效果是真不好，用这种方式提取出来的特征，真不如用pca效果好
  * 其次， 这种提特征的方式，感觉是可以用在gan中的，gan 中总是用一个概率分布来拟合一个分布，但是如果我们知道这个分布的特征，应该会更方便拟合。 但是这个效果还是不让我满意，我还不如用pca，
  * 另外一个方面，pca 可不是什么都能分析的，它又不能得到图像的特征，但是我也不是得到图像的特征，我想得到的是图像分布的特征，反正不好干，无所谓。
## 另外，我想用tensorflow 写一个svm，在此之前我先写了一个感知机
  * svm可不是那么好写的，看着简单，找一个超平面就好了。事实上，NIMA超平面可没那么好早。
  * 知道为什么svm 效果那么好吗，就是因为它那个支持向量啊， 我们目的其实也是去找这个向量，难就难在这
  * 另外，svm转化为二次规划问题的时候，这个对偶问题可不是那么容易解的，我想用smo, 但是我看了五遍没看到这个原理
  * 其实svm 就是实现一个分类，好的作用就是，线性分类。当然我们可以加一个核，但是，我甚至不知道我想要的图像是一个什么分布。（我心里是真没走出gan。 其实本来没什么爱好，但是做久了，感情就来了。就好像跟他谈了场恋爱，没结果，但是就是舍不得。）。事实上， 这个分布是一个无限的分布，我门不可能完全模拟出来，但是就和正太分布一样，大部分都在一个小范围内，其实gan也想干这个事， 把这个小范围找出来。其实，我觉得不可靠。
  * svm 效果好。但是一般情况下，感知机效果也不错啊。我假设，用低维的正太分布，去模拟图像的分布，交线是一个低维流形，这时候是不可能线性可分的。或者说，测度几乎处处为0。这时候，一个感知机，真的够了。你拟合的再好。你怎么都是低维的数据。你不可能有大量的拟合。所以。我用感知机也能得到一个比较好的结果。我还没试验。下次试试。（对了。这里的感知机我就是想替换判别器（gan））
## 学了tensorboard
* 说实话，是个不错的东西，我现在还只会，实时画一画误差。
* 我是自己编译的tensorflow， tensorboard 还需要另外编译，而且编译后执行命令要使用bazel-bin, 特别麻烦，然后我用alais 替换了原来的命令。
