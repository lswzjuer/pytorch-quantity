# Contribution 指南

## 代码风格规范

- **Google python coding style**: [Google 开源项目风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)
- **Python3 only**: [python3 features](https://github.com/arogozhnikov/python3_with_pleasure)
- **Pytorch style**: [PyTorch Styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)
- **层级换行要用4个空格**

## 代码实现规范

一次提交需要有三个部分。

- **主代码**
    - 划分为common（小组件）, components (抽象组件，必须基于小组件搭建)，task (任务)
    - **准则**
        - 如果发现要实现的组件有很多可以公用的小组件，那么放在抽象组件里，小组件放在common里。
        - 小组件尽可能函数化，无状态化，一个函数只做一件事。
        - 抽象组件是可选抽象，具体task可以自由组合抽象组件和小组件进行自己网络的搭建，不一定需要准守抽象组件的方式，以免限制太死。
- **测试代码**
    - 函数与类需要有对应的test，确保功能的正确性
    - test放在test目录下，与主代码目录划分一致，可以参考`test/common/io_test.py`, 用`assert`语句即可，函数前需要加一个test_, 写完后运行 `pytest test` 就可以进行测试。
- **文档**
    - 实现的函数与类要有对应的文档，说明功能，输入输出
    - 如果实现的东西比较上层，最好能说明一下设计的思路

## code review

1. 代码修改
    1. 新建自己的分支，修改完成后，自己在本地把自己的分支 merge 进 master 分支
    2. 也可以直接在master上修改，然后 git add, git commit
2. 代码在本地 master 上commit后，在master分支下输入 `arc diff --preview`，进入code review页面（**需要先在http://cr.fabu.ai 上创建一个账号，拿到API token**）
    - 因为task（比如lidar fusion）和roadtensor不在同一个项目下，所以如果修改了两者，需要分别在两个项目主目录进行commit，然后分别在各自目录`arc diff --preview`
3. 指定reviewer
4. 通过后，`arc land`就可以将代码提交