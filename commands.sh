#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${ROOT_DIR}/.vendor/ProtoMotions"
PYTHON_BIN="${PYTHON_BIN:-python}"

KEYPOINTS="${ROOT_DIR}/sample_data/230213_no_speak_001__A185_keypoints.npy"
RETARGET_DIR="${ROOT_DIR}/output/retargeted"
RETARGETED="${RETARGET_DIR}/230213_no_speak_001__A185_retargeted.npz"
PROTO_DIR="${ROOT_DIR}/output/proto"
PROTO_MOTION="${PROTO_DIR}/230213_no_speak_001__A185_retargeted.motion"
MEDIA_DIR="${ROOT_DIR}/docs/media"
USD_REL="protomotions/data/assets/Kangaroo/usd/kangaroo_grippers_ias/kangaroo_grippers_ias_configured.usd"

JOG_KEYPOINTS="${ROOT_DIR}/sample_data/231121_jog_ff_start_180_R_002__A531_keypoints.npy"
JOG_RETARGET_DIR="${ROOT_DIR}/output/jog/retargeted"
JOG_RETARGETED="${JOG_RETARGET_DIR}/231121_jog_ff_start_180_R_002__A531_retargeted.npz"
JOG_PROTO_DIR="${ROOT_DIR}/output/jog/proto"
JOG_PROTO_MOTION="${JOG_PROTO_DIR}/231121_jog_ff_start_180_R_002__A531_retargeted.motion"
JOG_MOTION_LIB="${ROOT_DIR}/output/motion_lib/kangaroo_jog_start.pt"

require_vendor() {
    if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
        echo "Missing ${VENDOR_DIR}. Run ./setup_upstream.sh first." >&2
        exit 1
    fi
}

require_keypoints() {
    if [[ ! -f "${KEYPOINTS}" ]]; then
        echo "Missing sample keypoints: ${KEYPOINTS}" >&2
        echo "Copy your SOMA23 keypoint NPY there or run extraction in ProtoMotions." >&2
        exit 1
    fi
}

require_jog_keypoints() {
    if [[ ! -f "${JOG_KEYPOINTS}" ]]; then
        echo "Missing Jog keypoints: ${JOG_KEYPOINTS}" >&2
        echo "Copy the extracted SOMA23 keypoint NPY into sample_data/." >&2
        exit 1
    fi
}

sync_overlay() {
    require_vendor
    cp -a "${ROOT_DIR}/overlay/." "${VENDOR_DIR}/"
}

run_vendor() {
    require_vendor
    (cd "${VENDOR_DIR}" && "${PYTHON_BIN}" "$@")
}

show_model() {
    sync_overlay
    run_vendor examples/load_kangaroo_xml_mujoco.py
}

show_mapping() {
    require_keypoints
    sync_overlay
    run_vendor examples/visualize_kangaroo_mapping_setup_mujoco.py \
        --keypoints "${KEYPOINTS}" --frame 0 --side-by-side
}

retarget() {
    require_keypoints
    sync_overlay
    mkdir -p "${RETARGET_DIR}"
    run_vendor data/scripts/retarget_soma_keypoints_to_kangaroo.py \
        "${KEYPOINTS}" "${RETARGETED}" --max-nfev 100
}

show_retargeted() {
    require_keypoints
    [[ -f "${RETARGETED}" ]] || { echo "Run ./commands.sh retarget first." >&2; exit 1; }
    sync_overlay
    run_vendor examples/visualize_kangaroo_retarget_mapping_mujoco.py \
        --keypoints "${KEYPOINTS}" \
        --retargeted "${RETARGETED}" \
        --in-place --side-by-side
}

render_mujoco_videos() {
    require_keypoints
    [[ -f "${RETARGETED}" ]] || { echo "Run ./commands.sh retarget first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${MEDIA_DIR}"
    run_vendor examples/render_kangaroo_retarget_videos_mujoco.py \
        --keypoints "${KEYPOINTS}" \
        --retargeted "${RETARGETED}" \
        --output-dir "${MEDIA_DIR}" \
        --prefix kangaroo_arm \
        --fps 30 --width 1280 --height 720 --static-seconds 5
}

convert_xml_to_usd() {
    sync_overlay
    run_vendor usd_convert/convert_kangaroo_ias.py
    run_vendor usd_convert/configure_kangaroo_ias_usd.py
    cp -a "${VENDOR_DIR}/${USD_REL}" "${ROOT_DIR}/overlay/${USD_REL}"
    echo "Updated repository USD: overlay/${USD_REL}"
}

test_usd() {
    sync_overlay
    run_vendor examples/visualize_kangaroo_ias_isaaclab.py --physics
}

convert_to_proto() {
    [[ -f "${RETARGETED}" ]] || { echo "Run ./commands.sh retarget first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${PROTO_DIR}"
    run_vendor data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
        --retargeted-motion-dir "${RETARGET_DIR}" \
        --output-dir "${PROTO_DIR}" \
        --robot-type kangaroo \
        --input-fps 30 --output-fps 30 --force-remake
}

show_isaaclab() {
    require_keypoints
    [[ -f "${PROTO_MOTION}" ]] || { echo "Run ./commands.sh convert first." >&2; exit 1; }
    sync_overlay
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${PROTO_MOTION}" \
        --robot kangaroo --simulator isaaclab \
        --hide-markers --in-place --camera-view front \
        --source-keypoints "${KEYPOINTS}" \
        --source-offset-y 1.5 --source-yaw-deg -90
}

record_isaaclab() {
    require_keypoints
    [[ -f "${PROTO_MOTION}" ]] || { echo "Run ./commands.sh convert first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${MEDIA_DIR}"
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${PROTO_MOTION}" \
        --robot kangaroo --simulator isaaclab \
        --hide-markers --in-place --camera-view front \
        --source-keypoints "${KEYPOINTS}" \
        --source-offset-y 1.5 --source-yaw-deg -90 \
        --record-video "${MEDIA_DIR}/kangaroo_arm_isaaclab.mp4" \
        --record-frames 112 --record-width 1280 --record-height 720
}

show_newton() {
    [[ -f "${PROTO_MOTION}" ]] || { echo "Run ./commands.sh convert first." >&2; exit 1; }
    sync_overlay
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${PROTO_MOTION}" \
        --robot kangaroo --simulator newton \
        --hide-markers --in-place --camera-view front
}

record_newton() {
    [[ -f "${PROTO_MOTION}" ]] || { echo "Run ./commands.sh convert first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${MEDIA_DIR}"
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${PROTO_MOTION}" \
        --robot kangaroo --simulator newton \
        --hide-markers --in-place --camera-view front \
        --record-video "${MEDIA_DIR}/kangaroo_arm_newton.mp4" \
        --record-frames 112
}

make_gifs() {
    mkdir -p "${MEDIA_DIR}"
    local name
    for name in kangaroo_arm_mapping_before kangaroo_arm_retargeted kangaroo_arm_isaaclab kangaroo_arm_newton
    do
        [[ -f "${MEDIA_DIR}/${name}.mp4" ]] || { echo "Missing ${name}.mp4" >&2; exit 1; }
        ffmpeg -y -i "${MEDIA_DIR}/${name}.mp4" \
            -filter_complex "[0:v]fps=15,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
            -loop 0 "${MEDIA_DIR}/${name}.gif"
    done
}

jog_retarget() {
    require_jog_keypoints
    sync_overlay
    mkdir -p "${JOG_RETARGET_DIR}"
    run_vendor data/scripts/retarget_soma_keypoints_to_kangaroo.py \
        "${JOG_KEYPOINTS}" "${JOG_RETARGETED}" --max-nfev 100
}

jog_view() {
    require_jog_keypoints
    [[ -f "${JOG_RETARGETED}" ]] || { echo "Run ./commands.sh jog-retarget first." >&2; exit 1; }
    sync_overlay
    run_vendor examples/visualize_kangaroo_retarget_mapping_mujoco.py \
        --keypoints "${JOG_KEYPOINTS}" --retargeted "${JOG_RETARGETED}" \
        --side-by-side
}

jog_convert() {
    [[ -f "${JOG_RETARGETED}" ]] || { echo "Run ./commands.sh jog-retarget first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${JOG_PROTO_DIR}"
    run_vendor data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
        --retargeted-motion-dir "${JOG_RETARGET_DIR}" \
        --output-dir "${JOG_PROTO_DIR}" --robot-type kangaroo \
        --input-fps 30 --output-fps 30 --force-remake
}

jog_package() {
    [[ -f "${JOG_PROTO_MOTION}" ]] || { echo "Run ./commands.sh jog-convert first." >&2; exit 1; }
    sync_overlay
    mkdir -p "$(dirname "${JOG_MOTION_LIB}")"
    run_vendor protomotions/components/motion_lib.py \
        --motion-path "${JOG_PROTO_MOTION}" \
        --output-file "${JOG_MOTION_LIB}" --device cpu
}

jog_isaaclab() {
    require_jog_keypoints
    [[ -f "${JOG_PROTO_MOTION}" ]] || { echo "Run ./commands.sh jog-convert first." >&2; exit 1; }
    sync_overlay
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${JOG_PROTO_MOTION}" --robot kangaroo \
        --simulator isaaclab --hide-markers --camera-view front --follow-camera \
        --source-keypoints "${JOG_KEYPOINTS}" \
        --source-offset-y 1.5 --source-yaw-deg -90
}

jog_newton() {
    [[ -f "${JOG_PROTO_MOTION}" ]] || { echo "Run ./commands.sh jog-convert first." >&2; exit 1; }
    sync_overlay
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${JOG_PROTO_MOTION}" --robot kangaroo \
        --simulator newton --hide-markers --camera-view front
}

jog_mujoco_video() {
    require_jog_keypoints
    [[ -f "${JOG_RETARGETED}" ]] || { echo "Run ./commands.sh jog-retarget first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${MEDIA_DIR}"
    run_vendor examples/render_kangaroo_retarget_videos_mujoco.py \
        --keypoints "${JOG_KEYPOINTS}" --retargeted "${JOG_RETARGETED}" \
        --output-dir "${MEDIA_DIR}" --prefix kangaroo_jog \
        --fps 30 --width 1280 --height 720 --no-in-place
    mv -f "${MEDIA_DIR}/kangaroo_jog_retargeted.mp4" \
        "${MEDIA_DIR}/kangaroo_jog_mujoco.mp4"
}

jog_isaac_video() {
    require_jog_keypoints
    [[ -f "${JOG_PROTO_MOTION}" ]] || { echo "Run ./commands.sh jog-convert first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${MEDIA_DIR}"
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${JOG_PROTO_MOTION}" --robot kangaroo \
        --simulator isaaclab --hide-markers --camera-view front --follow-camera \
        --source-keypoints "${JOG_KEYPOINTS}" \
        --source-offset-y 1.5 --source-yaw-deg -90 \
        --record-video "${MEDIA_DIR}/kangaroo_jog_isaaclab.mp4" \
        --record-frames 110 --record-width 1280 --record-height 720
}

jog_newton_video() {
    [[ -f "${JOG_PROTO_MOTION}" ]] || { echo "Run ./commands.sh jog-convert first." >&2; exit 1; }
    sync_overlay
    mkdir -p "${MEDIA_DIR}"
    run_vendor examples/motion_libs_visualizer.py \
        --motion_files "${JOG_PROTO_MOTION}" --robot kangaroo \
        --simulator newton --hide-markers --camera-view front \
        --record-video "${MEDIA_DIR}/kangaroo_jog_newton.mp4" --record-frames 110
}

jog_gifs() {
    local name
    for name in kangaroo_jog_mujoco kangaroo_jog_isaaclab kangaroo_jog_newton; do
        [[ -f "${MEDIA_DIR}/${name}.mp4" ]] || { echo "Missing ${name}.mp4" >&2; exit 1; }
        ffmpeg -y -i "${MEDIA_DIR}/${name}.mp4" \
            -filter_complex "[0:v]fps=15,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
            -loop 0 "${MEDIA_DIR}/${name}.gif"
    done
}

jog_train() {
    [[ -f "${JOG_MOTION_LIB}" ]] || { echo "Run ./commands.sh jog-package first." >&2; exit 1; }
    sync_overlay
    run_vendor protomotions/train_agent.py \
        --robot-name kangaroo --simulator isaaclab \
        --experiment-path examples/experiments/mimic/mlp.py \
        --experiment-name kangaroo_jog_mimic_20m \
        --motion-file "${JOG_MOTION_LIB}" \
        --num-envs 256 --batch-size 1024 --ngpu 1 --seed 0 \
        --training-max-steps 20000000 \
        --overrides agent.model.actor_optimizer.lr=5e-6
}

jog_pipeline() {
    jog_retarget
    jog_convert
    jog_package
}

usage() {
    cat <<'EOF'
Usage: ./commands.sh COMMAND

  sync          copy this repository's overlay into .vendor/ProtoMotions
  model         load the Kangaroo XML directly in MuJoCo
  mapping       show frame-0 semantic mapping without optimization
  retarget      optimize the complete sample motion
  view          show source skeleton and optimized robot in MuJoCo
  mujoco-videos render the two MuJoCo MP4 files
  usd           convert XML -> imported USD -> configured closed-loop USD
  usd-test      load the configured USD with PhysX in IsaacLab
  convert       convert the optimized NPZ to ProtoMotions .motion
  isaaclab      play the converted motion in IsaacLab
  isaac-video   record one IsaacLab loop as MP4
  newton        play the converted motion in Newton
  newton-video  record one front-view Newton loop as MP4
  gifs          convert all four MP4 files to README GIFs
  jog-retarget  optimize the complete SOMA23 jog-start clip
  jog-view      show the optimized jog and source skeleton in MuJoCo
  jog-convert   convert the jog NPZ to ProtoMotions .motion
  jog-package   package the jog .motion as a training MotionLib
  jog-isaaclab  play jog and source skeleton with a follow camera
  jog-newton    play the jog in Newton with its root-follow camera
  jog-mujoco-video render the MuJoCo jog comparison MP4
  jog-isaac-video record the IsaacLab jog MP4
  jog-newton-video record the Newton jog MP4
  jog-gifs      convert the three jog MP4s to README GIFs
  jog-train     start stable 20M-step IsaacLab Mimic training
  jog-pipeline  run retarget -> convert -> package
EOF
}

case "${1:-help}" in
    sync) sync_overlay ;;
    model) show_model ;;
    mapping) show_mapping ;;
    retarget) retarget ;;
    view) show_retargeted ;;
    mujoco-videos) render_mujoco_videos ;;
    usd) convert_xml_to_usd ;;
    usd-test) test_usd ;;
    convert) convert_to_proto ;;
    isaaclab) show_isaaclab ;;
    isaac-video) record_isaaclab ;;
    newton) show_newton ;;
    newton-video) record_newton ;;
    gifs) make_gifs ;;
    jog-retarget) jog_retarget ;;
    jog-view) jog_view ;;
    jog-convert) jog_convert ;;
    jog-package) jog_package ;;
    jog-isaaclab) jog_isaaclab ;;
    jog-newton) jog_newton ;;
    jog-mujoco-video) jog_mujoco_video ;;
    jog-isaac-video) jog_isaac_video ;;
    jog-newton-video) jog_newton_video ;;
    jog-gifs) jog_gifs ;;
    jog-train) jog_train ;;
    jog-pipeline) jog_pipeline ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
