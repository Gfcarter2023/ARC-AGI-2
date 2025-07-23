import os
import warnings
import matplotlib.pyplot as plt
from   matplotlib import colors
warnings.filterwarnings('ignore')

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


def get_path(name):
    return f'data\input\{name}' if os.path.exists(f'data\input\{name}') else name


def plot_task(task, task_solutions, i, t, size=2.5, w1=0.9, prediction=None):
    titleSize = 16
    num_train = len(task['train'])
    num_test = len(task['test'])

    wn = num_train + num_test
    fig, axs = plt.subplots(2, wn, figsize=(size * wn, 2 * size))
    plt.suptitle(f'Task #{i}, {t}', fontsize=titleSize, fontweight='bold', y=1, color='#eeeeee')

    '''train:'''
    for j in range(num_train):
        plot_one(axs[0, j], j, task, 'train', 'input', w=w1)
        plot_one(axs[1, j], j, task, 'train', 'output', w=w1)

    '''test:'''
    for k in range(num_test):
        plot_one(axs[0, j + k + 1], k, task, 'test', 'input', w=w1)
        task['test'][k]['output'] = task_solutions[k]
        plot_one(axs[1, j + k + 1], k, task, 'test', 'output', w=w1)

    axs[1, j + 1].set_xticklabels([])
    axs[1, j + 1].set_yticklabels([])
    axs[1, j + 1] = plt.figure(1).add_subplot(111)
    axs[1, j + 1].set_xlim([0, wn])

    '''Separators:'''
    colorSeparator = 'white'
    for m in range(1, wn):
        axs[1, j + 1].plot([m, m], [0, 1], '--', linewidth=1, color=colorSeparator)
    axs[1, j + 1].plot([num_train, num_train], [0, 1], '-', linewidth=3, color=colorSeparator)

    axs[1, j + 1].axis("off")

    '''Frame and background:'''
    fig.patch.set_linewidth(5)  # widthframe
    fig.patch.set_edgecolor('black')  # colorframe
    fig.patch.set_facecolor('#444444')  # background

    plt.tight_layout()

    print(f'#{i}, {t}')  # for fast and convinience search
    plt.show()

    if prediction is not None:
        print("Model's Solution:")  # Shows Predicted solution
        fig, axs = plt.subplots(1, 2, figsize=(size * 2, size))
        x = 0
        for attempt_name, attempt_grid in prediction[0].items():
            input_matrix = attempt_grid
            axs[x].imshow(input_matrix, cmap=cmap, norm=norm)

            # ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
            plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
            axs[x].set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
            axs[x].set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

            '''Grid:'''
            axs[x].grid(visible=True, which='both', color='#666666', linewidth=0.8)

            axs[x].tick_params(axis='both', color='none', length=0)

            '''sub title:'''
            axs[x].set_title(attempt_name, color='#dddddd')
            x += 1

        '''Frame and background:'''
        fig.patch.set_linewidth(5)  # widthframe
        fig.patch.set_edgecolor('black')  # colorframe
        fig.patch.set_facecolor('#444444')  # background

        plt.tight_layout()

    plt.show()


def plot_one(ax, i, task, train_or_test, input_or_output, solution=None, w=0.8):
    fs = 12
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    # ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

    '''Grid:'''
    ax.grid(visible=True, which='both', color='#666666', linewidth=w)

    ax.tick_params(axis='both', color='none', length=0)

    '''sub title:'''
    ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color='#dddddd')


def add_plot(loc, plot, attempt_name, axs, fig):
    axs[loc].imshow(plot, cmap=cmap, norm=norm)
    # ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    axs[loc].set_xticks([x - 0.5 for x in range(1 + len(plot[0]))])
    axs[loc].set_yticks([x - 0.5 for x in range(1 + len(plot))])

    '''Grid:'''
    axs[loc].grid(visible=True, which='both', color='#666666', linewidth=0.8)

    axs[loc].tick_params(axis='both', color='none', length=0)

    '''sub title:'''
    axs[loc].set_title(attempt_name, color='#dddddd')

    '''Frame and background:'''
    fig.patch.set_linewidth(5)  # widthframe
    fig.patch.set_edgecolor('black')  # colorframe
    fig.patch.set_facecolor('#444444')  # background

    plt.tight_layout()


def plot_incorrect(incorrect_prediction, size=2.5):
    print(incorrect_prediction['task_id'])
    print('Correct Size: ' + str(incorrect_prediction['correct_width']) + 'x' + str(
        incorrect_prediction['correct_height']))
    print('Predicted Size: ' + str(incorrect_prediction['predicted_width']) + 'x' + str(
        incorrect_prediction['predicted_height']))
    fig, axs = plt.subplots(1, 2, figsize=(size * 2, size))
    add_plot(0, incorrect_prediction['correct_output'], "Correct", axs, fig)
    add_plot(1, incorrect_prediction['predicted_output'], "Prediction", axs, fig)
    plt.show()
    