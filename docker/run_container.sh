docker rm -f catgrasp
CATGRASP_DIR=$(pwd)/../
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 \
-it --network=host \
--name catgrasp  \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-v $CATGRASP_DIR:/home/catgrasp \
-v ~/data/catgrasp/artifacts:/home/catgrasp/artifacts \
-v ~/data/catgrasp/urdf:/home/catgrasp/urdf \
-v ~/data/catgrasp/data:/home/catgrasp/data \
-v ~/data/catgrasp/dataset:/home/catgrasp/dataset \
-v ~/data/catgrasp/logs:/home/catgrasp/logs \
-v /tmp:/tmp  \
--ipc=host \
-e GIT_INDEX_FILE \
0ba8342fbfdb bash

