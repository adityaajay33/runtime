FROM ros:humble

#install system dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    libopencv-dev \
    python3-opencv \
    wget \
    && rm -rf /var/lib/apt/lists/*

#install onnx runtime (architecture-aware for x64 and arm64)
ARG TARGETARCH
RUN case "${TARGETARCH}" in \
        "amd64") ORT_ARCH=x64 ;; \
        "arm64") ORT_ARCH=aarch64 ;; \
        *) ORT_ARCH=x64 ;; \
    esac && \
    ORT_VER=1.16.3 && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-${ORT_ARCH}-${ORT_VER}.tgz" && \
    tar -zxvf "onnxruntime-linux-${ORT_ARCH}-${ORT_VER}.tgz" && \
    cp "onnxruntime-linux-${ORT_ARCH}-${ORT_VER}/include/"* /usr/local/include/ && \
    cp "onnxruntime-linux-${ORT_ARCH}-${ORT_VER}/lib/"libonnxruntime.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf "onnxruntime-linux-${ORT_ARCH}-${ORT_VER}.tgz" "onnxruntime-linux-${ORT_ARCH}-${ORT_VER}"

#create workspace
WORKDIR /ros2_ws/src/ptk

#copy source code
COPY . .

#build the package
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

#source workspace on container start
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
