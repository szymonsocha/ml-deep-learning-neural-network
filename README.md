# ml-deep-learning-neural-network
Comparison of neural network performanc with other ML methods for two different types of data (image classification and regression).

## Classification

### Dataset description

The dataset comes from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) contains 50000 photos of 43 different German road signs.

### Dataset preparation

At the beginning, we conducted an initial data exploration, which allowed us to conclude that we have an unbalanced dataset. In addition, we have presented sample photos available in the dataset to show that not all signs seem to be very obvious.

As part of the image preprocessing, we took the following steps:
- photos were rescaled to a resolution of 32x32
- converted to grayscale
- pixels were flattened and normalized to values between 0-1

Due to the lack of a validation set, it was decided to divide the training set in the proportion of 80\%/20\%. Thanks to this, we have obtained 3 independent datasets.
Then, depending on the algorithm used, we additionally transformed the dependent variable.

### Modelling part

We used 4 different classification algorithms for modelling:
- Random Forest
- LightGBM
- Convolutional Neural Network (based on LeNet-5 architecture)
- CNN with augmented data (for the best CNN architecture)

Accuracy was used as a performance metric, but a confusion matrix and a classification report were also built for each model.
In the case of the first two algorithms, Randomized Search Cross Validation was used to find the optimal hyperparameters.
For neural network, it was decided to use different convolution values and monitored the value of the loss function of the validation set.

### Results

|                |   Valid | Test |
|---------------:|-------:|----:|
| Random Forest  |  61% | 51% |
| LightGBM        | 86% | 68% |
| CNN (based on LeNet-5) |  97% | 87% |
| CNN with augmented data |  77% | 48% |

Data augmentation definitely decrease the overall performance, which may be due to:
- our model has simply too small capacity and it's not able to learn all the patterns in our new data
- despite the exclusion of flips, it is possible that there are other parameters that we have overlooked that cause us to create new road signs with a completely different meaning.

## Regression

### Data

#### Dataset Description

The dataset contains information about the specifications of laptops and their prices given in Euros. The dataset contains 1320 observations and 13 columns describing the specifications of the laptops:
- **Company** - *string* - Laptop Manufacturer
- **Product** - *string* - Brand and Model
- **TypeName** - *string* - Type (Notebook, Ultrabook, Gaming, etc.)
- **Inches** - *numeric* - Screen Size
- **ScreenResolution** - *string* - Screen Resolution
- **Cpu** - *string* - Central Processing Unit (CPU)
- **Ram** - *string* - Laptop RAM
- **Memory** - *string* - Hard Disk / SSD Memory
- **GPU** - *string* - Graphics Processing Units (GPU)
- **OpSys** - *string* - Operating System
- **Weight** - *string* - Laptop Weight
- **Price euros** - *numeric* - Price (Euro)
 
Dataset is available on Kaggle [here](https://www.kaggle.com/datasets/muhammetvarl/laptop-price).

#### Data preparation

In order to prepare the data, we perform **EDA**. We also perform **feature engineering**:
- We extract the screen type and resolution from the `ScreenResolution` column
- From the `GPU` column, we extract the processor model and processor clocking
- From the `Memory` column, we extract whether the laptop has two drives or one, what type and what capacity

We note that the variable `Price euros` is right-skewed. We logarithmize this variable to give it a distribution closer to the normal distribution.
After feature engineering, we divide the dataset into a training set and a test set. We use the test dataset later **only** for performance comparison of finished models.

On the training set, we remove one **outlier**. Originally in the dataset there is one laptop with 64GB RAM, which looks like a very powerful, expensive, gaming laptop. We remove this observation.

### Algorithms considered:

* **Random Forest**
* XGBoost
* Neural Network

To tune hyperparameters, we perform **Randomized Search Cross Validation** for each algorithm. Additionally, for Neural Network we standarize values at each fold (in order to **prevent data leakage**) with a use of [scaler](https://github.com/szymonsocha/ml-deep-learning-neural-network/blob/main/regression/scaler/sc_nn.pkl). Later, we use the same scaler on the test set.

### Results

|                |   RMSE | MAE | $R^2[\\%]$ |
|---------------:|-------:|----:|-----------:|
| Random Forest  |  43031 | 132 |      93.26 |
| XGBoost        | 102681 | 166 |      91.63 |
| Neural Network |  91711 | 194 |      86.55 |

With the lowest RMSE and MAE and the highest $R^2$ score, Random Forest is the best performing model.


#### Predicted vs. Real values

|     |   Random Forest |   XGBoost |   Neural Network |   Real Values |
|----:|----------------:|----------:|-----------------:|--------------:|
|   0 |            1247 |      1364 |             1264 |          1672 |
|   1 |            1144 |      1237 |             1132 |          1149 |
|   2 |             521 |       448 |              392 |           499 |
|   3 |             899 |       825 |              899 |           899 |
|   4 |            1484 |      1312 |             1222 |          1244 |
|   5 |            1434 |      1286 |             1069 |          1399 |
|   6 |             732 |       752 |              928 |           719 |
|   7 |             471 |       432 |              390 |           459 |
|   8 |            1431 |      1491 |             1542 |          1191 |
|   9 |             345 |       361 |              295 |           349 |
|  10 |            2097 |      2316 |             2790 |          1899 |
|  11 |             720 |       673 |              629 |           726 |
|  12 |            1821 |      1654 |             1792 |          1813 |
|  13 |             806 |       584 |              719 |           806 |
|  14 |            1895 |      2226 |             1823 |          1949 |
|  15 |            1022 |       970 |              934 |           899 |
|  16 |            1307 |      1508 |             1331 |          1145 |
|  17 |             958 |      1039 |             1033 |           961 |
|  18 |             705 |       718 |              944 |           713 |
|  19 |            1901 |      1906 |             2472 |          1649 |
|  20 |            2482 |      2039 |             2769 |          2349 |
|  21 |             720 |       657 |              690 |           720 |
|  22 |            1106 |       998 |             1236 |           798 |
|  23 |             311 |       277 |              304 |           265 |
|  24 |            1018 |      1161 |             1166 |          1149 |
|  25 |             899 |       942 |              780 |           999 |
|  26 |             815 |       871 |              786 |           959 |
|  27 |             718 |       782 |              734 |           825 |
|  28 |            1601 |      1789 |             1671 |          2277 |
|  29 |            1189 |      1186 |             1113 |          1179 |
|  30 |             459 |       439 |              480 |           459 |
|  31 |            1582 |      1644 |             1693 |          1725 |
|  32 |            1606 |      1677 |             1805 |          1969 |
|  33 |             974 |      1032 |             1008 |           959 |
|  34 |            1168 |      1011 |             1154 |           943 |
|  35 |             726 |       781 |              713 |           745 |
|  36 |             712 |       672 |              710 |           659 |
|  37 |             473 |       531 |              452 |           399 |
|  38 |            1218 |      1240 |             1268 |          1229 |
|  39 |             263 |       259 |              313 |           330 |
|  40 |            1710 |      1969 |             1899 |          1868 |
|  41 |             581 |       513 |              527 |           519 |
|  42 |             673 |       796 |              622 |           739 |
|  43 |            2431 |      1910 |             2585 |          2290 |
|  44 |             617 |       542 |              578 |           547 |
|  45 |            1221 |      1235 |             1087 |          1377 |
|  46 |            1647 |      1883 |             1827 |          1983 |
|  47 |             233 |       258 |              228 |           279 |
|  48 |             757 |       885 |              850 |           795 |
|  49 |            1379 |      1380 |             1437 |          1458 |
|  50 |            1799 |      1573 |             1571 |          1499 |
|  51 |             504 |       499 |              560 |           629 |
|  52 |             959 |       973 |              921 |          1199 |
|  53 |            1275 |      1186 |              864 |          1100 |
|  54 |             569 |       624 |              537 |           557 |
|  55 |            1233 |      1189 |             1081 |          1026 |
|  56 |             233 |       258 |              228 |           249 |
|  57 |             899 |       903 |              975 |           819 |
|  58 |            1199 |       913 |              928 |           941 |
|  59 |             899 |       825 |              899 |           899 |
|  60 |            1552 |      1361 |             1630 |          1749 |
|  61 |             925 |       922 |              917 |           841 |
|  62 |             434 |       440 |              497 |           404 |
|  63 |             703 |       580 |              609 |           682 |
|  64 |             286 |       273 |              279 |           299 |
|  65 |            1916 |      2208 |             1863 |          2050 |
|  66 |             539 |       481 |              457 |           529 |
|  67 |             837 |       756 |              875 |           949 |
|  68 |            1079 |      1017 |             1012 |          1094 |
|  69 |            2251 |      1983 |             2570 |          1279 |
|  70 |            2627 |      2364 |             3267 |          2799 |
|  71 |             895 |      1008 |              790 |          1119 |
|  72 |             583 |       650 |              728 |           530 |
|  73 |             876 |       891 |              740 |           806 |
|  74 |             346 |       330 |              370 |           309 |
|  75 |             429 |       387 |              380 |           469 |
|  76 |             759 |       787 |              755 |           639 |
|  77 |            1661 |      1650 |             1180 |          1993 |
|  78 |            1305 |      1217 |             1212 |          1599 |
|  79 |            1508 |      1568 |             1757 |          1499 |
|  80 |            1150 |      1093 |             1259 |          1399 |
|  81 |             674 |       729 |              966 |           615 |
|  82 |             406 |       350 |              326 |           349 |
|  83 |             735 |       793 |              777 |           714 |
|  84 |             293 |       277 |              279 |           278 |
|  85 |            1327 |      1660 |             1533 |          2051 |
|  86 |             464 |       516 |              444 |           439 |
|  87 |            2552 |      2675 |             2124 |          2999 |
|  88 |            1019 |       939 |              968 |           989 |
|  89 |             926 |       877 |              947 |           819 |
|  90 |            1570 |      1943 |             1832 |          1510 |
|  91 |            2867 |      2090 |             2771 |          3240 |
|  92 |             889 |       807 |              958 |           809 |
|  93 |             788 |       828 |              674 |           726 |
|  94 |            1141 |      1101 |             1075 |          1474 |
|  95 |             451 |       434 |              459 |           426 |
|  96 |            1022 |       908 |              916 |           859 |
|  97 |             647 |       756 |              769 |           716 |
|  98 |             644 |       706 |              642 |           784 |
|  99 |            1006 |      1049 |             1072 |          1049 |
| 100 |             307 |       318 |              293 |           330 |
| 101 |             569 |       522 |              506 |           549 |
| 102 |             624 |       731 |              601 |           665 |
| 103 |            1910 |      1808 |             2002 |          2041 |
| 104 |            1021 |      1269 |             1146 |          1000 |
| 105 |             806 |       584 |              719 |           806 |
| 106 |             886 |      1014 |              906 |          1169 |
| 107 |             609 |       643 |              899 |           499 |
| 108 |            1391 |      1357 |             1425 |          1097 |
| 109 |             420 |       453 |              702 |           299 |
| 110 |            1873 |      1963 |             1545 |          1799 |
| 111 |            1337 |      1329 |             1101 |          1427 |
| 112 |             953 |       863 |             1075 |           879 |
| 113 |            2767 |      2638 |             2829 |          2729 |
| 114 |             292 |       267 |              271 |           299 |
| 115 |            2249 |      2123 |             1717 |          2449 |
| 116 |            1392 |      1422 |             1378 |          1199 |
| 117 |            1706 |      1933 |             1958 |          1449 |
| 118 |             882 |       862 |              792 |           979 |
| 119 |             704 |       711 |              667 |           759 |
| 120 |            5315 |      3414 |             5051 |          6099 |
| 121 |            1077 |      1103 |              946 |          1124 |
| 122 |             800 |       830 |              750 |           797 |
| 123 |            1261 |      1425 |             1749 |          1299 |
| 124 |            1073 |       921 |             1021 |           789 |
| 125 |             470 |       531 |              466 |           466 |
| 126 |            1337 |      1328 |             1111 |          1099 |
| 127 |             570 |       618 |              751 |           649 |
| 128 |            1518 |      1854 |             1755 |          1379 |
| 129 |            1151 |      1165 |             1522 |           699 |
| 130 |            1338 |      1436 |             1487 |          1271 |
