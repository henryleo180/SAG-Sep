#!/bin/bash

set -eo pipefail

BPC_DIR=$(cd "$(dirname "$0")" && pwd)

die() { echo "setup_routeopt.sh: $*" >&2; exit 2; }

# --- which application -----------------------------------------------------
APP=cvrp
while [ $# -gt 0 ]; do
    case "$1" in
        --app)
            [ $# -ge 2 ] || die "--app needs an argument (cvrp or vrptw)"
            APP=$2
            shift 2 ;;
        --app=*)
            APP=${1#--app=}
            shift ;;
        -h|--help)
            echo "usage: $0 [--app cvrp|vrptw]"
            exit 0 ;;
        *)
            die "unknown argument \"$1\"; usage: $0 [--app cvrp|vrptw]" ;;
    esac
done

case "$APP" in
    cvrp)  APP_TYPE=CVRP;  BINARY_NAME=cvrp ;;
    vrptw) APP_TYPE=VRPTW; BINARY_NAME=vrptw ;;
    *) die "unknown application \"$APP\"; expected cvrp or vrptw" ;;
esac

case "$APP" in
    vrptw) VRPTW_TYPE=${SAGSEP_VRPTW_TYPE:-TW_MAIN_RESOURCE} ;;
    *)     VRPTW_TYPE=SINGLE_RESOURCE ;;
esac
case "$VRPTW_TYPE" in
    SINGLE_RESOURCE|TW_MAIN_RESOURCE|CAP_MAIN_RESOURCE) ;;
    *) die "SAGSEP_VRPTW_TYPE must be SINGLE_RESOURCE, TW_MAIN_RESOURCE or CAP_MAIN_RESOURCE (got \"$VRPTW_TYPE\")" ;;
esac

if [ -n "${SAGSEP_BPC_MODULES:-}" ]; then
    eval "${SAGSEP_BPC_MODULES}"
fi

SAGSEP_C=${SAGSEP_C:-$BPC_DIR/../../package/c}
ROUTEOPT_DIR=$BPC_DIR/RouteOpt
CVRP_DIR=$ROUTEOPT_DIR/packages/application/cvrp
ENGINE_BUILD=$BPC_DIR/build/sagsep
SAGSEP_LIB=$ENGINE_BUILD/libsagsep.a
APP_BUILD=$BPC_DIR/build/$APP
BINARY=$CVRP_DIR/bin/$BINARY_NAME
BUILD_JOBS=${SAGSEP_BPC_JOBS:-4}

# --- preconditions ---------------------------------------------------------
[ -n "${ONNXRUNTIME_HOME:-}" ] || die "ONNXRUNTIME_HOME is not set"
[ -f "$ONNXRUNTIME_HOME/include/onnxruntime_c_api.h" ] || \
    die "no include/onnxruntime_c_api.h under ONNXRUNTIME_HOME=$ONNXRUNTIME_HOME"
[ -f "$ONNXRUNTIME_HOME/lib/libonnxruntime.so" ] || \
    die "no lib/libonnxruntime.so under ONNXRUNTIME_HOME=$ONNXRUNTIME_HOME"
[ -n "${GUROBI_HOME:-}" ] || die "GUROBI_HOME is not set"
[ -f "$GUROBI_HOME/include/gurobi_c.h" ] || \
    die "no include/gurobi_c.h under GUROBI_HOME=$GUROBI_HOME"
[ -f "$SAGSEP_C/include/sep_api.h" ] || \
    die "no include/sep_api.h under SAGSEP_C=$SAGSEP_C"
SAGSEP_C=$(cd "$SAGSEP_C" && pwd)

# --- 1. the vendored, already patched RouteOpt ------------------------------
[ -f "$ROUTEOPT_DIR/packages/rounded_cap_cuts/separation/src/generate_rcc.cpp" ] || \
    die "no RouteOpt tree at $ROUTEOPT_DIR"
grep -Fq 'SAGSEP_RCC_POOL' \
    "$ROUTEOPT_DIR/packages/rounded_cap_cuts/separation/src/generate_rcc.cpp" || \
    die "the vendored RouteOpt is missing patch 0001: no SAGSEP_RCC_POOL in generate_rcc.cpp; apply patches/0001-sagsep-rcc-pool.patch."
grep -Fq 'SAGSEP_APP_TYPE' \
    "$CVRP_DIR/src/main/include/cvrp_macro.hpp" || \
    die "the vendored RouteOpt is missing patch 0002: no SAGSEP_APP_TYPE in cvrp_macro.hpp; apply patches/0002-app-type-switch.patch."
grep -Fq 'SAGSEP_VRPTW_TYPE' \
    "$CVRP_DIR/src/main/include/cvrp_macro.hpp" || \
    die "the vendored RouteOpt is missing patch 0003: no SAGSEP_VRPTW_TYPE in cvrp_macro.hpp; apply patches/0003-bpc-root-and-vrptw-mode.patch."

# --- 2. the SAG-Sep engine as a static library ------------------------------
echo "==> building libsagsep.a"
mkdir -p "$ENGINE_BUILD"
make -C "$SAGSEP_C" lib \
    GUROBI_HOME="$GUROBI_HOME" \
    ONNXRUNTIME_HOME="$ONNXRUNTIME_HOME" \
    BUILD_DIR="$ENGINE_BUILD/obj" \
    LIB="$SAGSEP_LIB" \
    -j"$BUILD_JOBS"
[ -f "$SAGSEP_LIB" ] || die "the engine build produced no $SAGSEP_LIB"

# --- 3. RouteOpt's own bootstrap -------------------------------------------
if [ ! -f "$ROUTEOPT_DIR/packages/external/xgb/lib/libxgboost.so" ] || \
   [ ! -f "$CVRP_DIR/lib/hgs/lib/libhgscvrp.so" ] || \
   [ ! -f "$ROUTEOPT_DIR/packages/external/cvrpsep/obj/libcvrpsep.a" ]; then
    echo "==> running RouteOpt's build.py bootstrap (XGBoost, HGS, CVRPSEP)"
    ( cd "$ROUTEOPT_DIR" && printf '%s\n' "$GUROBI_HOME" | python3 build.py )
fi
[ -f "$ROUTEOPT_DIR/packages/external/cvrpsep/obj/libcvrpsep.a" ] || \
    die "CVRPSEP was not built (packages/external/cvrpsep/obj/libcvrpsep.a is missing)"

# --- 4. the application -----------------------------------------------------
if [ "$APP" = vrptw ]; then
    echo "==> configuring and building the $APP_TYPE application (labeling mode $VRPTW_TYPE)"
else
    echo "==> configuring and building the $APP_TYPE application"
fi
cmake -S "$CVRP_DIR" -B "$APP_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSAGSEP_APP_TYPE="$APP_TYPE" \
    -DSAGSEP_VRPTW_TYPE="$VRPTW_TYPE" \
    -DSAGSEP_C="$SAGSEP_C" \
    -DSAGSEP_LIB="$SAGSEP_LIB" \
    -DONNXRUNTIME_HOME="$ONNXRUNTIME_HOME"
cmake --build "$APP_BUILD" --target cvrp -j"$BUILD_JOBS"

[ -x "$BINARY" ] || die "no binary at $BINARY"
echo
echo "built $BINARY"
ls -l "$BINARY"
echo
echo "run it with:  $BPC_DIR/$APP/run_bpc.sh <cvrpsep|sagseptts> <instance> <upper-bound>"
