#!/bin/bash
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root" 
  exit 1
fi
echo "Begin upgrading..."
rm -f /etc/apt/sources.list.d/graphics-drivers-ubuntu-ppa-*
add-apt-repository -y ppa:graphics-drivers
sed -i 's/http:\/\/ppa.launchpad.net/https:\/\/launchpad.proxy.ustclug.org/g' /etc/apt/sources.list.d/graphics-drivers-ubuntu-ppa-*.list
apt update
release=$(lsb_release -sr)
if (( $(echo $release'<='16.04 | bc -l) )); then
  apt install -y nvidia-410;
else
  apt install -y nvidia-driver-410 libnvidia-gl-410 nvidia-dkms-410 nvidia-kernel-source-410 libnvidia-compute-410 nvidia-compute-utils-410 libnvidia-decode-410 libnvidia-encode-410 nvidia-utils-410 libnvidia-ifr1-410;
fi

echo "If no error occurs, you might now reboot to bring new driver into effect"
