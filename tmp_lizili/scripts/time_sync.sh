# cn.pool.ntp.org

echo "time ntp server: " $1
grep -q ntpdate /etc/crontab

if [ $? -eq 1 ]; then
    echo "*/1 * * * * root ntpdate -v -u $1" | sudo tee -a /etc/crontab
fi

# ntpdate running log at /var/log/syslog

sudo ntpdate -v -u $1
