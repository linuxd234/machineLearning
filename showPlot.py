#########################################################################
# 功能：绘制训练及验证损失曲线
# 2020/9/28 create by linuxd

#########################################################################
import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    # plot绘制
    plt.title('train_history')

    plt.ylabel(train)
    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='upper left')
    # legend是给图加上图例，什么颜色是什么曲线
    plt.show()
