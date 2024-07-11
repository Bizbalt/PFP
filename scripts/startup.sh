#!/bin/bash
echo "ENTER STARTUP"
logrotate -vf /etc/logrotate.conf
# Redirect stdout to a log file
exec > >(tee /var/log/container.log)
# Redirect stderr to a different error log file and also to stderr
exec 2> >(tee /var/log/container.log >&2)

# Activate the Conda environment
echo "ENTER PFP"
source activate pfp

# create a "pfp" directory
mkdir -p pfp
cd pfp

# copy example dir
cp -rn ../examples/ ./

# print the current working directory
pwd
ls -al

# start jupyter server

echo "ENTER JUPYTER"
jupyter lab --ip 0.0.0.0 --no-browser --allow-root

