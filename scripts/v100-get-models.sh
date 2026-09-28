#!/usr/bin/env bash
# 下载 V100 实测用的两个模型，并把草稿模型转成 F16：
#   目标模型 unsloth/Qwen3.8-27B-GGUF 的 Q8_0（约 29 GB）
#   草稿模型 z-lab/Qwen3.8-27B-DFlash2-GGUF 的 BF16（约 3.9 GB）
# V100（sm_70）没有 BF16 硬件支持，所以草稿模型要转 F16。
# 用法：scripts/v100-get-models.sh [-s modelscope|hf] [-d 目录] [-q llama-quantize路径] [-h]

set -euo pipefail

MODEL_REPO="unsloth/Qwen3.8-27B-GGUF"
MODEL_FILE="Qwen3.8-27B-Q8_0.gguf"
DRAFT_REPO="z-lab/Qwen3.8-27B-DFlash2-GGUF"
DRAFT_FILE="Qwen3.8-27B-DFlash2-BF16.gguf"
DRAFT_F16_FILE="Qwen3.8-27B-DFlash2-F16.gguf"

SOURCE="modelscope"
MODEL_DIR="./models"
QUANTIZE="./build/bin/llama-quantize"

usage() {
    cat <<EOF
用法：$(basename "$0") [-s modelscope|hf] [-d 目录] [-q llama-quantize路径] [-h]

  -s  下载源：modelscope（默认，国内推荐）或 hf（Hugging Face，基址取自 HF_ENDPOINT）
  -d  模型目录，默认 ./models（不存在会创建）
  -q  llama-quantize 路径，默认 ./build/bin/llama-quantize
  -h  显示本帮助并退出
EOF
}

die() {
    printf "%s\n" "$@" >&2
    exit 1
}

resolve_url() {
    local repo="$1"
    local file="$2"

    if [ "$SOURCE" = "modelscope" ]; then
        echo "https://modelscope.cn/models/$repo/resolve/master/$file"
    else
        echo "${HF_ENDPOINT:-https://huggingface.co}/$repo/resolve/main/$file"
    fi
}

download() {
    local file="$1"
    local url="$2"
    local path="$MODEL_DIR/$file"

    if [ -f "$path" ]; then
        echo "$file 已存在，跳过"
        return
    fi

    echo "下载 $file ..."
    local try
    for try in 1 2 3 4 5 6 7 8 9 10; do
        if curl -L --fail --retry 5 --retry-delay 5 --progress-bar -C - -o "$path.part" "$url"; then
            mv "$path.part" "$path"
            echo "$file 下载完成"
            return
        fi
        echo "下载中断，10 秒后从断点继续（第 $try 次）" >&2
        sleep 10
    done
    die "$file 下载失败；重新运行本脚本会从断点继续"
}

while getopts ":s:d:q:h" opt; do
    case "$opt" in
        s) SOURCE="$OPTARG" ;;
        d) MODEL_DIR="$OPTARG" ;;
        q) QUANTIZE="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) usage >&2; exit 1 ;;
        :) usage >&2; exit 1 ;;
    esac
done

case "$SOURCE" in
    modelscope|hf) ;;
    *) die "未知下载源：$SOURCE（可选 modelscope 或 hf）" ;;
esac

command -v curl >/dev/null 2>&1 || die "未找到 curl，请先安装"

mkdir -p "$MODEL_DIR"

download "$MODEL_FILE" "$(resolve_url "$MODEL_REPO" "$MODEL_FILE")"
download "$DRAFT_FILE" "$(resolve_url "$DRAFT_REPO" "$DRAFT_FILE")"

if [ -f "$MODEL_DIR/$DRAFT_F16_FILE" ]; then
    echo "$DRAFT_F16_FILE 已存在，跳过转换"
else
    [ -x "$QUANTIZE" ] || die "找不到可执行的 llama-quantize：$QUANTIZE（请先按 README 编译本项目）"
    echo "转换 $DRAFT_FILE -> $DRAFT_F16_FILE ..."
    if ! "$QUANTIZE" "$MODEL_DIR/$DRAFT_FILE" "$MODEL_DIR/$DRAFT_F16_FILE" F16; then
        rm -f "$MODEL_DIR/$DRAFT_F16_FILE"
        die "转换失败，已删除不完整的 $DRAFT_F16_FILE"
    fi
    echo "转换完成"
fi

cat <<EOF

全部完成。

启动示例：

CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server \\
  -m $MODEL_DIR/$MODEL_FILE -ngl 999 \\
  --split-mode tensor --tensor-split 1,1,1,1 \\
  --model-draft $MODEL_DIR/$DRAFT_F16_FILE \\
  --spec-type draft-dflash --spec-draft-n-max 7 \\
  ...（其余参数见 README 的“推荐启动参数”）

转换用的 $DRAFT_FILE 已保留，确认可用后可以删除。
EOF
