## usage: ./wait_until_end.sh 161556 your_next_command arg1 arg2

pid=$1  # First argument is the PID
shift  # Shift the arguments to remove the first argument (PID)

while ps -p $pid > /dev/null; do sleep 1; done; "$@"