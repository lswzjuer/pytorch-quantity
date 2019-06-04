sudo rm -rf /etc/X11/xorg.conf
sudo nvidia-xconfig -a --allow-empty-initial-configuration --use-display-device="DFP-0" --connected-monitor="DFP-0"
printf '[SeatDefaults]\ndisplay-setup-script=xhost +local:' | sudo tee /etc/lightdm/lightdm.conf.d/xhost.conf
sudo /etc/init.d/lightdm restart
