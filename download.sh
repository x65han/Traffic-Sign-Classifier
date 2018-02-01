# This script downloads the data sets and puts the pickle file in data_sets
# run `chmod 741 download.sh` then run `./download.sh`
# Author: Johnson Han

# define variables
zip_file_name=traffic-signs-data.zip
folder_name=data_sets
url="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/$zip_file_name"

# clean up directory
rm *.p
rm -rf *.zip
rm -rf $folder_name

# download & unzip zip file
wget $url
unzip $zip_file_name

# process zip file
mkdir $folder_name
mv *.p $folder_name

# clean up raw files
rm -rf *.zip
