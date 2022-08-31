mkdir data
cd data
echo "Downloading the TransPIR dataset ..."
wget "https://cloud.tsinghua.edu.cn/f/2feaea15e9094941b4bd/?dl=1"
mv 'index.html?dl=1' data.zip
echo "Start unzipping ..."
unzip data.zip
echo "TransPIR dataset is ready!"
