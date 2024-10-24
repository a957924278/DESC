import os
import numpy as np
import matplotlib.pyplot as plt
from desc.backend import jnp, execute_on_cpu
from desc.compute.data_index import register_compute_fun
from desc.compute import compute as compute_fun, get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.io import IOAble
from desc.optimizable import Optimizable, optimizable_parameter
from desc.utils import errorif, Timer, setdefault
from desc.objectives.objective_funs import _Objective
from desc.vmec import VMECIO

def rescale_factor(eq, L_new=1.7, B_new=5.7):
    """计算等离子体平衡的尺寸和磁场强度缩放因子。"""
    grid_L = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=1.0, sym=True)
    grid_B = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=0, sym=True)
    
    L_old = eq.compute("a", grid=grid_L)["a"]
    B_old = np.mean(eq.compute("|B|", grid=grid_B)["|B|"])
    
    return L_new / L_old, B_new / B_old

def create_simple_in(sbeg, ntestpart, wout_path, cL, cB):
    """创建simple.in文件"""
    content = f"""&config
  trace_time = 1d-1
  sbeg = {sbeg}d0
  ntestpart = {ntestpart}
  netcdffile = '{wout_path}'
  vmec_B_scale = {cB}
  vmec_RZ_scale = {cL}
  /
  """
    with open("simple.in", "w") as file:
        file.write(content)

def generate_label(eq, sbeg):
    """生成标签"""
    if eq.compute("p")["p"][0] < 1e-12:
        return f"s{sbeg}_vac"
    beta = eq.compute("<beta>_vol")["<beta>_vol"] * 100
    return f"s{sbeg}_beta{beta:.2f}"

def plot_alpha_loss_fraction(eq, wout_path, sbeg=0.01, ntestpart=500, ax=None, label=None, color=None, linestyle='-', figsize=(5, 4), dpi=200,verbose=0):
    """计算并绘制alpha损失分数。"""
    plt.rcParams.update({"font.size": 18})
    
    label = label or generate_label(eq, sbeg)
    
    if not os.path.exists(f"{label}.dat"):
        cL, cB = rescale_factor(eq)
        create_simple_in(sbeg, ntestpart, wout_path, cL, cB)
        
        if not os.path.exists(wout_path):
            print(f"\n将eq保存为{wout_path}文件...")
            VMECIO.save(eq=eq, path=wout_path, verbose=0, surfs=128)
            print(f"{wout_path}保存完毕\n")
        
        print(f"\n正在计算{label}的Alpha损失分数...")
        if verbose == 0:
            os.system("simple.x > /dev/null 2>&1")
        else:
            os.system("simple.x")
        print(f"{label}的Alpha损失分数计算完成。\n")
        
        os.system(f"mv confined_fraction.dat {label}.dat")
    else:
        print(f"{label}.dat已存在，跳过损失分数的计算。")
    
    data_rescaled = np.loadtxt(f"{label}.dat", comments="#", dtype=np.float64)
    t, loss_fraction_rescaled = data_rescaled[:, 0], 1 - data_rescaled[:, 1] - data_rescaled[:, 2]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    ax.plot(t, loss_fraction_rescaled, label=label, color=color, linestyle=linestyle)
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    return fig, ax

def read_mercier(file_path):
    """读取Mercier数据"""
    data = []
    separator_count = 0
    start_reading = False

    with open(file_path, 'r') as file:
        for line in file:
            if '-------' in line:
                separator_count += 1
                if separator_count == 2:
                    start_reading = True
                continue

            if start_reading:
                values = line.strip().split()
                if values:  # 确保行不为空
                    data.append([float(val) for val in values])

    data = np.array(data)
    
    return data
