<p align="center"><img width="40%" src="./assests/logo.jpg" /></p>

--------------------------------------------------------------------------------
## Requirements
* Tensorflow 1.8
* Python 3.6

```python
> python download.py celebA
```

## Usage
### Downloading the dataset
```python
> python download.py celebA
```

```
├── dataset
   └── celebA
       ├── train
           ├── 000001.jpg 
           ├── 000002.jpg
           └── ...
       ├── test
           ├── a.jpg (The test image that you wanted)
           ├── b.png
           └── ...
       ├── list_attr_celeba.txt (for attribute information) 
```

### Train
* python main.py --phase train

### Test
* python main.py --phase test 

### Pretrained model
* Download [celebA_checkpoint](https://drive.google.com/open?id=1ezwtU1O_rxgNXgJaHcAynVX8KjMt0Ua-)

## Summary
![overview](./assests/overview.PNG)

## Results
### Women
![women](./assests/women.png)

### Men
![men](./assests/men.png)

## Author
Junho Kim
