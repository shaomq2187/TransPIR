cd exps
cd fixed_cameras
echo "Downloading the pretrained models ..."
wget "https://cloud.tsinghua.edu.cn/f/1b90d24c43e641c29b38/?dl=1"
mv 'index.html?dl=1' fixed_cameras.zip
echo "Start unzipping ..."
unzip fixed_cameras.zip
echo "Pretrained models is ready!"
