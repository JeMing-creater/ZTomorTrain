# 广肿 MR 数据实验

# Before use:
You need to check data first! 
Please setup `config.yml`.`data_check` and run code:<br>
```
python data_check.py
```

Or use code from 

`https://github.com/JeMing-creater/ZTomorTest`

And then you will get two txts(NonsurgicalMR.txt, SurgicalMR.txt), which are used for loading data.
<br>
you can set data in every dir you like, and change sub-dir'name like `NonsurgicalMR` and `SurgicalMR`.

## get project
```
git clone https://github.com/JeMing-creater/ZTumorTrain.git
```

## requirements
```
cd requirements

# Mamba
cd Mamba/causal-conv1d
python setup.py install
cd Mamba/mamba
python setup.py install
# Mamba sample setting
# Find the mamba_sample.py file and replace it with requirements\mamba_sample.py

cd requirements
pip install -r requirements.txt
```

## training
single device training
```
python3 main.py
```
multi-devices training
```
sh run.sh
```