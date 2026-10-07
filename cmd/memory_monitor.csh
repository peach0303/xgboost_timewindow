#!/bin/csh -f
if ( $#argv < 1 ) then
    echo "Usage: csh memory_monitor.csh <target_pid> [threshold_gb=100] [log_file=/tmp/memory_monitor.log]"
    exit 2
endif
set TARGET_PID = "$1"
set THRESHOLD_GB = 100
if ( $#argv >= 2 ) set THRESHOLD_GB = "$2"
set LOG_FILE = /tmp/memory_monitor.log
if ( $#argv >= 3 ) set LOG_FILE = "$3"
while ( 1 )
    ps -p "$TARGET_PID" >& /dev/null
    if ( $status != 0 ) exit 0
    set avail_gb = `awk '/^MemAvailable:/ {printf "%d", $2/1024/1024}' /proc/meminfo`
    echo "`date '+%Y-%m-%d %H:%M:%S'`,${TARGET_PID},${avail_gb}GB" >> "$LOG_FILE"
    if ( $avail_gb < $THRESHOLD_GB ) then
        echo "Memory below threshold: ${avail_gb}GB < ${THRESHOLD_GB}GB; killing ${TARGET_PID}" >> "$LOG_FILE"
        kill "$TARGET_PID"
        exit 1
    endif
    sleep 10
end
