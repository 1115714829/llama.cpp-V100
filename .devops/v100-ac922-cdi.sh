#!/bin/bash
# Generate a CDI spec that passes the NVIDIA GPUs into podman containers on an IBM Power AC922
# (ppc64le). The NVIDIA Container Toolkit is not published for ppc64le, so the spec is written
# directly: the GPU device nodes, plus the host driver libraries mounted under their soname.
#
#   sudo bash .devops/v100-ac922-cdi.sh          # writes /etc/cdi/nvidia.yaml
#   podman run --device nvidia.com/gpu=all ...   # all GPUs
#   podman run --device nvidia.com/gpu=0 ...     # one GPU
#
# Run it again after a driver update (the library versions are part of the spec).
set -euo pipefail

OUT=${1:-/etc/cdi/nvidia.yaml}
LIBDIR=${LIBDIR:-/usr/lib64}

drv=$(ls "$LIBDIR"/libcuda.so.*.* 2>/dev/null | sed -E 's/.*libcuda\.so\.//' | sort -V | tail -1)
[ -n "$drv" ] || { echo "no libcuda.so.<version> in $LIBDIR" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"

gpus=$(ls /dev/nvidia[0-9]* 2>/dev/null | sed 's#/dev/nvidia##' | sort -n)
[ -n "$gpus" ] || { echo "no /dev/nvidia<N> device nodes" >&2; exit 1; }

{
    echo "cdiVersion: 0.5.0"
    echo "kind: nvidia.com/gpu"
    echo "devices:"
    for i in $gpus; do
        echo "  - name: \"$i\""
        echo "    containerEdits:"
        echo "      deviceNodes:"
        echo "        - path: /dev/nvidia$i"
    done
    echo "  - name: all"
    echo "    containerEdits:"
    echo "      deviceNodes:"
    for i in $gpus; do
        echo "        - path: /dev/nvidia$i"
    done
    echo "containerEdits:"
    echo "  deviceNodes:"
    for n in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia-caps/nvidia-cap1 /dev/nvidia-caps/nvidia-cap2; do
        [ -e "$n" ] && echo "    - path: $n"
    done
    echo "  mounts:"
    # each driver library is mounted under its soname, so no symlinks are needed in the container
    for lib in libcuda libnvidia-ml libnvidia-ptxjitcompiler libnvidia-nvvm libnvidia-allocator libnvidia-cfg libnvidia-gpucomp; do
        src="$LIBDIR/$lib.so.$drv"
        [ -e "$src" ] || continue
        soname=$lib.so.1
        [ "$lib" = libnvidia-nvvm ] && soname=$lib.so.4
        echo "    - hostPath: $src"
        echo "      containerPath: $LIBDIR/$soname"
        echo "      options: [ro, nosuid, nodev, bind]"
    done
    if [ -x /usr/bin/nvidia-smi ]; then
        echo "    - hostPath: /usr/bin/nvidia-smi"
        echo "      containerPath: /usr/bin/nvidia-smi"
        echo "      options: [ro, nosuid, nodev, bind]"
    fi
} > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "wrote $OUT (driver $drv, GPUs: $(echo $gpus | tr '\n' ' '))"
