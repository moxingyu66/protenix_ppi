#!/usr/bin/env bash

set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_output_dir="${script_dir}/../artifacts/env"
output_dir="${1:-${default_output_dir}}"
mkdir -p "${output_dir}"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_file="${output_dir}/environment_linux_${timestamp}.txt"

run_section() {
    local title="$1"
    shift
    printf '\n===== %s =====\n' "${title}"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1 || printf 'COMMAND FAILED (exit %s)\n' "$?"
    else
        printf 'UNAVAILABLE: %s not found\n' "$1"
    fi
}

{
    printf 'Protenix-PPI G0 environment report\n'
    printf 'Collected: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    printf 'Review and redact hostnames, usernames, IP addresses, private paths, scheduler accounts, and credentials before sharing.\n'

    run_section "Operating system" uname -a
    run_section "OS release" cat /etc/os-release
    run_section "NVIDIA summary" nvidia-smi
    run_section "NVIDIA inventory" nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    run_section "GPU topology" nvidia-smi topo -m
    run_section "CPU" lscpu
    run_section "Memory" free -h
    run_section "Filesystem capacity" df -h
    run_section "Python" python3 --version
    run_section "Docker" docker --version
    run_section "Apptainer" apptainer --version
    run_section "Git" git --version
    run_section "Slurm partitions" sinfo
} > "${output_file}"

printf 'Environment report written to: %s\n' "${output_file}"

