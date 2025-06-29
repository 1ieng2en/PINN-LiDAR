# auto_plot.py
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import clear_output, display

def show_figures(shell):
    figs = [plt.figure(num) for num in plt.get_fignums()]
    if not figs:
        return
        
    n_figs = len(figs)
    w = 12 / 1.1
    h = w * (3 / 10)
    
    # 清除当前输出
    clear_output(wait=True)
    
    if n_figs == 1:
        figs[0].set_size_inches(w, h)
        display(figs[0])
    elif n_figs == 2:
        # 创建新的两栏布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h))
        
        # 复制图形内容
        for old_fig, new_ax in zip(figs, (ax1, ax2)):
            old_ax = old_fig.get_axes()[0]
            # 复制数据和设置
            for line in old_ax.get_lines():
                new_ax.plot(line.get_xdata(), line.get_ydata())
            new_ax.set_title(old_ax.get_title())
        
        plt.close('all')
        plt.tight_layout()
        display(fig)

# 注册回调
ip = get_ipython()
if ip is not None:
    ip.events.register('post_run_cell', show_figures)