# Vision-based Navigation Using Deep Reinforcement Learning
Official implementation of A2CAT-VN. A reinforcement learning architecture capable of navigating an agent, e.g. a mobile robot, to a target given by an image.
It extends the batched A2C algorithm with auxiliary tasks designed to improve visual navigation performance.

[Paper](https://arxiv.org/pdf/1908.03627.pdf)&nbsp;&nbsp;&nbsp;
[Web](https://jkulhanek.github.io/a2cat-vn)
 
<br>

# Getting started
Before getting started, ensure, that you have Python 3.6+ ready.
We recommend activating a new virtual environment for the repository:
```bash
python -m venv a2catvn-env
source a2catvn-env/bin/activate
```

Start by cloning this repository and installing the dependencies:
```bash
git clone https://github.com/jkulhanek/a2cat-vn.git
cd a2cat-vn
pip install -r requirements.txt
```

For discrete AI2THOR experiments, you can speed up the loading of the dataset by downloading the pre-computed dataset:
```bash
mkdir -p ~/.cache/visual-navigation/datasets
for package in thor-cached-212 thor-cached-208 thor-cached-218 thor-cached-225 thor-cached-212-174 thor-cached-208-174 thor-cached-218-174 thor-cached-225-174; do
    curl -L -o ~/.cache/visual-navigation/datasets/$package.pkl https://data.ciirc.cvut.cz/public/projects/2019VisionBasedNavigation/resources/$package.pkl
done
```

> **_NOTE:_**  SUNCG dataset is not longer available and we cannot provide dataset samples.

# Training
In order to start the training, run the following command:
```bash
python train.py {trainer}
```
where `{trainer}` is the name of the experiment and can be one of the following:
- `thor-cached-auxiliary`
- `cthor-multigoal-auxiliary`
- `chouse-auxiliary-superviised` (requires SUNCG dataset which is no longer publicly available!)
- `chouse16-auxiliary` (requires SUNCG dataset which is no longer publicly available!)

For `chouse*` experiments, you need to have House3D simulator installed and SUNCG dataset downloaded.
We recommend using provided docker image.