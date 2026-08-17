#!/bin/bash
# Build Umi OS to wasmdist/<name>/  (usage: ./build.sh <name>)
# ABI / EXPORT_NAME(createSim) / 出力名(sim.js) は hanabi_emu_cpp・taki_emu_cpp と共通。
set -e; cd "$(dirname "$0")"
EMSDK="${EMSDK:-/c/prog/emsdk/emsdk}"
export EM_CONFIG="$EMSDK/.emscripten"
export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$PATH"
NAME="${1:-umi_os}"
mkdir -p "wasmdist/$NAME"
em++ -O3 -std=c++17 -msimd128 -I. "src/$NAME.cpp" \
  -sMODULARIZE=1 -sEXPORT_NAME=createSim -sENVIRONMENT=web -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS=_sim_init,_sim_w,_sim_h,_sim_reset,_sim_step,_sim_render,_sim_click,_sim_set,_sim_action,_sim_get,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=cwrap,HEAPU8 \
  -o "wasmdist/$NAME/sim.js"
echo "built wasmdist/$NAME/sim.js (+.wasm)"
