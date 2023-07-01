#!/bin/sh

source setenv.sh

HOSTNAME=$(hostname)
DATE=$(date)

echo "Hello World at ${HOSTNAME} (${DATE})"
