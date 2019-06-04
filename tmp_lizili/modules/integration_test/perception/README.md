First config the bag param files. There are some example config xml files under /roadstar/modules/integration_test/perception/scripts/param/.

Second cp the label_data from "/nfs/lidar_data/labeled_data/" path. And then make sure the map relationship  between the labeled_data and the raw_data is correct.

Third go to file path of "/roadstar/modules/integration_test/perception/scripts/". Run cmd like this:
./test.sh param/example.xml

Finally wait the script to finish its run for seconds and cd the report path you config in  example.xml to take a look at the report file.

Any problems,please contact with kongxilong@fabu.ai.
