# this is mylearning of tensorflow scipy,numpy,matplotlib
我用scipy_lecture上的教程。其实上面还有python 的教程，已经做得很棒了。
## cmd+?可以多行注释
## numpy数组的二元运算，可以不是相同维度的，他们是通过扩展到相同维度进行的

## 原来numpy里面还有多项式操作，这仿造matlab简直一流啊



# scipy

## scipy.io
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
io.loadmat/io.savemat
from scipy import misc
msic.imread()

## scipy.linalg :linear algebra operations

## 切片操作是能取到到开头，但是取不到结尾。比如b[1:]能取到b[1]，但是b[:-1]是取不到最后一个的。
>>>>>>> origin/first
