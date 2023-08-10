
pip install --upgrade pip  # ensures that pip is current

pip install -U scikit-learn
pip install --user -U nltk

apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
python3 -m pip install konlpy
apt-get install curl git
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .

pip install evaluate
pip install bert-score

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mecab

wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip