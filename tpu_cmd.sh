#! /bin/bash

function _tpu_ips {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    gcloud alpha compute tpus tpu-vm describe $tpu_name --zone $tpu_zone --project $tpu_project | grep -oP 'externalIp: \K(.+)$' |   if [[ -n "$EXCLUDE_IP" ]]; then     grep -vE "$(echo "$EXCLUDE_IP" | tr ',' '|')";   else     cat;   fi

}

function _tpu_cp_ssh_key {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/google_compute_engine

    gcloud alpha compute tpus tpu-vm scp \
        $HOME/.ssh/authorized_keys \
        $tpu_name:/home/$USER/.ssh/ \
        --worker=all \
        --project $tpu_project \
        --zone $tpu_zone

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh-keygen -R "$host" 2>/dev/null
        ssh-keyscan -H "$host" >> ~/.ssh/known_hosts
    done

}

function _tpu_setup {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    _tpu_copy $tpu_zone $tpu_project $tpu_name

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host "cd ${PROJECT_HOME}/${PROJECT_NAME}; bash tpu_setup.sh" &
    done
    wait &> /dev/null
}

function _tpu_check {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        echo "============== Checking host: $host =============="
        ssh $host 'tmux capture-pane -pt launch -S -10'
        echo
        echo
    done
}

function _tpu_copy {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        rsync -avPI --exclude="${PROEJCT_NAME}/__pycache__" --exclude="${PROJECT_NAME}/.git" --exclude="${PROJECT_NAME}/output" --exclude="${PROJECT_NAME}/pretrained"  $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    done
    wait &> /dev/null
    sleep 1s

    #for host in $tpu_ips; do
    #    rsync -avPI --exclude="${PROJECT_HOME}/${PROEJCT_NAME}/__pycache__" --exclude="${PROJECT_HOME}/${PROJECT_NAME}/.git" --exclude="${PROJECT_HOME}/${PROJECT_NAME}/output" $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    #done
    #wait &> /dev/null
    #sleep 1s
}

function _tpu_stop {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host 'tmux kill-session -t launch ; pkill -9 python' &
    done
    wait &> /dev/null
}

function _tpu_launch {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    _tpu_copy $tpu_zone $tpu_project $tpu_name

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        # ensure files are synced
        ssh $host "tmux new -d -s launch ~/$PROJECT_NAME/tools/launch.sh" &
    done
    wait &> /dev/null
}

function _tpu_maintain {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    gcloud alpha compute tpus tpu-vm simulate-maintenance-event $tpu_name \
        --project $tpu_project \
        --zone=$tpu_zone \
        --workers=all
}

function _tpu_ssh {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    command="$4"

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host "$command" &
    done
    wait &> /dev/null
}

function _tpu_reboot {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host 'sudo reboot' &
    done
    wait &> /dev/null
}


function tpu {
    trap "trap - SIGINT SIGTERM; return 1;" SIGINT SIGTERM


    # =============== TPU Project Specific Definitions ===============
    export PROJECT_HOME="$HOME"
    export PROJECT_NAME='seq-jax'
    set -a; source "${PROJECT_HOME}/${PROJECT_NAME}/.env"; set +a
    tpu_zone='us-central2-b'
    if [ -n "$TPU_PROJECT" ] && [ "$1" = "$TPU_PROJECT" ]; then
        tpu_project="$TPU_PROJECT"
        tpu_zone='us-central2-b'
        tpu_gen='v4'
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    # =============== End of TPU Project Specific Definitions ===============


    if [ "$2" = "list" ]; then
        gcloud alpha compute tpus tpu-vm list --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "describe" ]; then
        gcloud alpha compute tpus tpu-vm describe $3 --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "ips" ]; then
        _tpu_ips $tpu_zone $tpu_project $3
    elif [ "$2" = "delete" ]; then
        gcloud alpha compute tpus tpu-vm delete $3 --zone $tpu_zone --project $tpu_project --quiet --async
    elif [ "$2" = "delete_queued" ]; then
        gcloud alpha compute tpus queued-resources delete $3 --project $tpu_project --zone $tpu_zone
    elif [ "$2" = "create" ]; then
        gcloud alpha compute tpus tpu-vm create $3 --zone $tpu_zone --project $tpu_project --accelerator-type=$4 --runtime-version=tpu-ubuntu2204-base
    elif [ "$2" = "create_spot" ]; then
        gcloud alpha compute tpus tpu-vm create $3 --zone $tpu_zone --project $tpu_project --accelerator-type=$4 --runtime-version=tpu-ubuntu2204-base --spot
    elif [ "$2" = "create_queued" ]; then
        gcloud alpha compute tpus queued-resources create $3 --node-id=$3 --zone=$tpu_zone --accelerator-type=$4 --runtime-version=tpu-ubuntu2204-base --spot
    elif [ "$2" = "cp_ssh_key" ]; then
        _tpu_cp_ssh_key $tpu_zone $tpu_project $3
    elif [ "$2" = "cs" ]; then
        # create
        sleep 90s
        _tpu_setup $tpu_zone $tpu_project $4
    elif [ "$2" = "check" ]; then
        _tpu_check $tpu_zone $tpu_project $3
    elif [ "$2" = "setup" ]; then
        _tpu_setup $tpu_zone $tpu_project $3
    elif [ "$2" = "copy" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
    elif [ "$2" = "stop" ]; then
        _tpu_stop $tpu_zone $tpu_project $3
    elif [ "$2" = "launch" ]; then
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "cl" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "maintain" ]; then
        _tpu_maintain $tpu_zone $tpu_project $3
    elif [ "$2" = "ssh" ]; then
        _tpu_ssh $tpu_zone $tpu_project $3 "$4"
    elif [ "$2" = "reboot" ]; then
        _tpu_reboot $tpu_zone $tpu_project $3
    elif [ "$2" = "df" ]; then
        _tpu_ssh $tpu_zone $tpu_project $3 'df -h | grep root'
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    trap - SIGINT SIGTERM
}


export -f tpu _tpu_ips _tpu_setup _tpu_check _tpu_copy _tpu_stop _tpu_launch _tpu_maintain _tpu_ssh _tpu_reboot
