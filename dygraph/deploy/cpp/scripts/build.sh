arch=`uname -m`
# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON
# 使用MKL or openblas
WITH_MKL=ON
# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF
# TensorRT 的路径，如果需要集成TensorRT，需修改为您实际安装的TensorRT路径
TENSORRT_DIR=$(pwd)/TensorRT/
# Paddle 预测库路径, 请修改为您实际安装的预测库路径
PADDLE_DIR=$(pwd)/paddle_inference
# Paddle 的预测库是否使用静态库来编译
# 使用TensorRT时，Paddle的预测库通常为动态库
WITH_STATIC_LIB=OFF
# 是否加密
WITH_ENCRYPTION=OFF
# OPENSSL 路径
OPENSSL_DIR=$(pwd)/deps/openssl-1.1.0k
# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda/lib64
# CUDNN 的 lib 路径
if [ $arch = "aarch64" ]; then
  WITH_MKL=OFF
  WITH_STATIC_LIB=OFF
  CUDNN_LIB=/usr/lib/aarch64-linux-gnu
else
  CUDNN_LIB=/usr/lib/x86_64-linux-gnu
  {
    bash $(pwd)/scripts/bootstrap.sh # 下载预编译版本的opencv依赖库
  } || {
    echo "Fail to execute script/bootstrap.sh"
    exit -1
  }
fi

# OPENCV 路径, 如果使用自带预编译版本可不修改
OPENCV_DIR=$(pwd)/deps/opencv3.4.6gcc4.8ffmpeg/

# 以下无需改动
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_ENCRYPTION=${WITH_ENCRYPTION} \
    -DOPENSSL_DIR=${OPENSSL_DIR}
make -j16
