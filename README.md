# VideoX - Multi-modal Video Content Understanding

***This is a collection of our video understanding work***

> [**MS-2D-TAN**](./MS-2D-TAN) (```@TPAMI'21```): **Multi-Scale 2D Temporal Adjacent Networks for Moment Localization with Natural Language**

> [**2D-TAN**](./2D-TAN) (```@AAAI'20```): **Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language**

## News

- :sunny: Hiring research interns: houwen.peng@microsoft.com
- :boom: Oct, 2021: Code for [**MS-2D-TAN**](./MS-2D-TAN) is now released.
- :boom: Sep, 2021: [**MS-2D-TAN**](./MS-2D-TAN) was accepted to TPAMI'21
- :boom: Dec, 2019: Code for [**2D-TAN**](./2D-TAN) is now released.
- :boom: Nov, 2019: [**2D-TAN**](./2D-TAN) was accepted to AAAI'20

## Works


### [MS-2D-TAN](./MS-2D-TAN)

In this paper, we study the problem of moment localization with natural language, and propose a extend our previous proposed 2D-TAN method to a multi-scale version. The core idea is to retrieve a moment from two-dimensional temporal maps at different temporal scales, which considers adjacent moment candidates as the temporal context. The extended version is capable of encoding adjacent temporal relation at different scales, while learning discriminative features for matching video moments with referring expressions. Our model is simple in design and achieves competitive performance in comparison with the state-of-the-art methods on three benchmark datasets.

<div align="center">
    <img width="70%" alt="MS-2D-TAN overview" src="./MS-2D-TAN/pipeline.jpg"/>
</div>

### [2D-TAN](./2D-TAN)

In this paper, we study the problem of moment localization with natural language, and propose a novel 2D Temporal Adjacent Networks(2D-TAN) method. The core idea is to retrieve a moment on a two-dimensional temporal map, which considers adjacent moment candidates as the temporal context. 2D-TAN is capable of encoding adjacent temporal relation, while learning discriminative feature for matching video moments with referring expressions. Our model is simple in design and achieves competitive performance in comparison with the state-of-the-art methods on three benchmark datasets.

<div align="center">
    <img width="80%" alt="2D-TAN overview" src="./2D-TAN/imgs/pipeline.jpg"/>
</div>

## Bibtex

```bibtex
@InProceedings{Zhang2021MS2DTAN,
    author = {Zhang, Songyang and Peng, Houwen and Fu, Jianlong and Lu, Yijuan and Luo, Jiebo},
    title = {Multi-Scale 2D Temporal Adjacent Networks for Moment Localization with Natural Language},
    booktitle = {TPAMI},
    year = {2021}
}


@InProceedings{2DTAN_2020_AAAI,
    author = {Zhang, Songyang and Peng, Houwen and Fu, Jianlong and Luo, Jiebo},
    title = {Learning 2D Temporal Adjacent Networks forMoment Localization with Natural Language},
    booktitle = {AAAI},
    year = {2020}
}
```

## License

License under an MIT license.
