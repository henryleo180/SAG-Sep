set -eo pipefail

die() { echo "run_bpc.sh: $*" >&2; exit 2; }

abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$(pwd)" "$1" ;;
    esac
}

check_instance_name() {
    local name=$1
    [ -n "$name" ] || die "SAGSEP_INSTANCE_NAME is empty"
    [ "${#name}" -lt 128 ] || die "SAGSEP_INSTANCE_NAME is longer than 127 bytes: $name"
    case "$name" in
        *[!\ -~]*) die "SAGSEP_INSTANCE_NAME must be printable ASCII, got \"$name\"" ;;
    esac
}

case "${BPC_APP:-}" in
    cvrp)  BPC_BINARY_NAME=cvrp;  BPC_INSTANCE_KIND="instance.vrp" ;;
    vrptw) BPC_BINARY_NAME=vrptw; BPC_INSTANCE_KIND="instance.txt" ;;
    *) die "run_bpc_common.sh was sourced with BPC_APP=\"${BPC_APP:-}\"; expected cvrp or vrptw" ;;
esac

if [ -n "${SAGSEP_BPC_MODULES:-}" ]; then
    eval "${SAGSEP_BPC_MODULES}"
fi

POOL=${1:-}
INSTANCE=${2:-}
UPPER_BOUND=${3:-}

if [ -z "$POOL" ] || [ -z "$INSTANCE" ] || [ -z "$UPPER_BOUND" ]; then
    echo "usage: $0 <cvrpsep|sagseptts> <$BPC_INSTANCE_KIND> <upper-bound>" >&2
    exit 2
fi

case "$POOL" in
    cvrpsep|sagseptts) ;;
    *) die "unknown RCC pool \"$POOL\"; expected cvrpsep or sagseptts" ;;
esac

case "$UPPER_BOUND" in
    ''|*[!0-9.eE+-]*) die "the upper bound must be a number, got \"$UPPER_BOUND\"" ;;
esac
awk -v v="$UPPER_BOUND" 'BEGIN { exit !(v + 0 == v && v + 0 > 0) }' </dev/null \
    || die "the upper bound must be a positive finite number, got \"$UPPER_BOUND\""

CVRP_DIR=$BPC_COMMON_DIR/RouteOpt/packages/application/cvrp
BINARY=$(abspath "${SAGSEP_BPC_BINARY:-$CVRP_DIR/bin/$BPC_BINARY_NAME}")
[ -x "$BINARY" ] || die "no RouteOpt binary at $BINARY -- run ../setup_routeopt.sh --app $BPC_APP first"
[ -f "$INSTANCE" ] || die "no such instance file: $INSTANCE"
INSTANCE_PATH=$(cd "$(dirname "$INSTANCE")" && pwd)/$(basename "$INSTANCE")

export SAGSEP_RCC_POOL=$POOL

if [ "$POOL" = "sagseptts" ]; then
    SAGSEP_MODEL=$(abspath "${SAGSEP_MODEL:-$BPC_COMMON_DIR/../../models/cvrp-spf-best.onnx}")
    [ -f "$SAGSEP_MODEL" ] || die "no such model file: $SAGSEP_MODEL"
    export SAGSEP_MODEL

    SAGSEP_CROSS_DEPOT=${SAGSEP_CROSS_DEPOT:-0}
    case "$SAGSEP_CROSS_DEPOT" in
        0|1) ;;
        *) die "SAGSEP_CROSS_DEPOT must be 0 or 1, got \"$SAGSEP_CROSS_DEPOT\"" ;;
    esac
    export SAGSEP_CROSS_DEPOT

    if [ "${SAGSEP_ALLOW_MODEL_MISMATCH:-}" != "1" ]; then
        case "$(basename "$SAGSEP_MODEL")" in
            cvrp-spf-best.onnx)
                [ "$SAGSEP_CROSS_DEPOT" = "0" ] || die \
                    "cvrp-spf-best was fine-tuned with cross_depot disabled; it wants" \
                    "SAGSEP_CROSS_DEPOT=0, not $SAGSEP_CROSS_DEPOT." \
                    "Set SAGSEP_ALLOW_MODEL_MISMATCH=1 to run it anyway." ;;
            cvrp-two-index-best.onnx)
                [ "$SAGSEP_CROSS_DEPOT" = "1" ] || die \
                    "cvrp-two-index-best was trained with cross_depot enabled; it wants" \
                    "SAGSEP_CROSS_DEPOT=1, not $SAGSEP_CROSS_DEPOT." \
                    "Set SAGSEP_ALLOW_MODEL_MISMATCH=1 to run it anyway." ;;
        esac
    fi

    export SAGSEP_PROVIDER=${SAGSEP_PROVIDER:-cpu}

    export SAGSEP_INSTANCE_FILE=$INSTANCE_PATH
    if [ -n "${SAGSEP_INSTANCE_NAME:-}" ]; then
        check_instance_name "$SAGSEP_INSTANCE_NAME"
        export SAGSEP_INSTANCE_NAME
    fi

    if [ -z "${SAGSEP_PRESET:-}" ]; then
        die "SAGSEP_PRESET is required: a size preset (n50 ... n1000) or a *.toml config file"
    else
        case "$SAGSEP_PRESET" in
            *.toml)
                SAGSEP_PRESET=$(abspath "$SAGSEP_PRESET")
                [ -f "$SAGSEP_PRESET" ] || die "no such preset file: $SAGSEP_PRESET" ;;
        esac
    fi
    export SAGSEP_PRESET
    if [ -n "${SAGSEP_ARENA_MIB:-}" ]; then export SAGSEP_ARENA_MIB; fi
fi

BANNER=
if [ "$POOL" = "sagseptts" ]; then
    BANNER=" preset=$SAGSEP_PRESET provider=$SAGSEP_PROVIDER"
    BANNER="$BANNER model=$SAGSEP_MODEL cross_depot=$SAGSEP_CROSS_DEPOT"
fi
echo "[RUN_BPC] app=$BPC_APP pool=$POOL instance=$INSTANCE_PATH ub=$UPPER_BOUND$BANNER"

cd "$CVRP_DIR"
exec "$BINARY" "$INSTANCE_PATH" -u "$UPPER_BOUND"
