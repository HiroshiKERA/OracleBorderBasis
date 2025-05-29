from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import math
import numpy as np


def plot_order_ideals(degree_bounds: List[int], 
                     order_ideals: List[List[Tuple[int, ...]]],
                     filename: Optional[str] = None,
                     track_changes: bool = False):
    """
    Plot multiple order ideals, automatically choosing 2D or 3D based on dimension
    
    Parameters:
    -----------
    degree_bounds : List[int]
        Maximum degree for each variable. Length must be 2 or 3
    order_ideals : List[List[Tuple[int, ...]]]
        List of order ideals to plot
    filename : Optional[str]
        If provided, save plot to this file
    track_changes : bool
        If True, highlight new points in each step with red color
        
    Raises:
    -------
    ValueError
        If degree_bounds length is not 2 or 3
    """
    dim = len(degree_bounds)
    if dim == 2:
        plot_order_ideals_2d(degree_bounds, order_ideals, filename, track_changes)
    elif dim == 3:
        plot_order_ideals_3d(degree_bounds, order_ideals, filename, track_changes)
    else:
        raise ValueError(f"Can only plot in 2D or 3D, got dimension {dim}")

def plot_order_ideal(degree_bounds: List[int], 
                    order_ideal: List[Tuple[int, ...]],
                    filename: Optional[str] = None):
    """
    Plot a single order ideal, automatically choosing 2D or 3D based on dimension
    
    Parameters:
    -----------
    degree_bounds : List[int]
        Maximum degree for each variable. Length must be 2 or 3
    order_ideal : List[Tuple[int, ...]]
        Order ideal to plot
    filename : Optional[str]
        If provided, save plot to this file
        
    Raises:
    -------
    ValueError
        If degree_bounds length is not 2 or 3
    """
    plot_order_ideals(degree_bounds, [order_ideal], filename)

import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Optional

def plot_order_ideals_2d(degree_bounds: List[int], 
                     order_ideals: List[List[Tuple[int, int]]], 
                     filename: Optional[str] = None,
                     track_changes: bool = False):
    """
    Plot multiple order ideals in a grid layout
    
    Parameters:
    -----------
    degree_bounds : List[int]
        Maximum degree for each variable [deg(x), deg(y)]
    order_ideals : List[List[Tuple[int, int]]]
        List of order ideals to plot
    filename : Optional[str]
        If provided, save plot to this file
    track_changes : bool
        If True, highlight new points in each step with red color
    """
    aspect_ratio = 1
    n_plots = len(order_ideals)
    n_cols = int(min(5 / aspect_ratio, n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    
    fig = plt.figure(figsize=(3*n_cols, 2*n_rows))
    
    base_color = '#1f77b4'  # Steel blue
    new_point_color = '#d62728'  # Red
    
    for idx, order_ideal in enumerate(order_ideals):
        ax = fig.add_subplot(n_rows, n_cols, idx+1)
        
        # Plot all possible grid points
        points = [(i,j) for i in range(degree_bounds[0] + 1) 
                 for j in range(degree_bounds[1] + 1)]
        x_coords, y_coords = zip(*points)
        ax.scatter(x_coords, y_coords, c='lightgray', s=20, zorder=2)
        
        # Plot order ideal points
        order_ideal_x, order_ideal_y = zip(*order_ideal)
        
        if track_changes and idx > 0:
            # Split points into existing and new
            prev_points = set(order_ideals[idx-1])
            current_points = set(order_ideal)
            new_points = current_points - prev_points
            
            # Plot existing points in blue
            existing_points = [(x, y) for x, y in order_ideal if (x, y) not in new_points]
            if existing_points:
                ex_x, ex_y = zip(*existing_points)
                ax.scatter(ex_x, ex_y, c=base_color, s=20, alpha=0.7, zorder=3, 
                          label='Existing points')
            
            # Plot new points in red
            if new_points:
                new_x, new_y = zip(*new_points)
                ax.scatter(new_x, new_y, c=new_point_color, s=20, alpha=0.7, zorder=4,
                          label='New points')
                # ax.legend(loc='upper right', fontsize='small')
        else:
            # Plot all points in blue
            ax.scatter(order_ideal_x, order_ideal_y, c=base_color, s=20, 
                      alpha=0.7, zorder=3)
        
        # Add grid lines
        for i in range(degree_bounds[0] + 1):
            ax.axvline(i, color='gray', alpha=0.2, zorder=1)
        for i in range(degree_bounds[1] + 1):
            ax.axhline(i, color='gray', alpha=0.2, zorder=1)
        
        # Set ticks and labels
        xticks = range(0, degree_bounds[0] + 1, max(1, degree_bounds[0] // 4))
        yticks = range(0, degree_bounds[1] + 1, max(1, degree_bounds[1] // 4))
        
        ax.set_xlabel('deg(x)')
        ax.set_ylabel('deg(y)')
        ax.grid(True, alpha=0.2, zorder=1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        if track_changes:
            ax.set_title(f'Step {idx+1}', pad=3)
        else:
            ax.set_title(f'Order Ideal #{idx+1}', pad=3)
        ax.set_box_aspect(aspect_ratio)
        
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_order_ideals_3d(degree_bounds: List[int], 
                        order_ideals: List[List[Tuple[int, int, int]]], 
                        filename: Optional[str] = None,
                        track_changes: bool = False):
    """
    Plot multiple 3D order ideals in a grid layout
    
    Parameters:
    -----------
    degree_bounds : List[int]
        Maximum degree for each variable [deg(x), deg(y), deg(z)]
    order_ideals : List[List[Tuple[int, int, int]]]
        List of order ideals to plot
    filename : Optional[str]
        If provided, save plot to this file
    track_changes : bool
        If True, highlight new points in each step with red color
    """
    n_plots = len(order_ideals)
    n_cols = int(min(5, n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    
    fig = plt.figure(figsize=(5*n_cols, 6*n_rows))
    
    base_color = '#1f77b4'  # Steel blue
    new_point_color = '#d62728'  # Red
    
    for idx, order_ideal in enumerate(order_ideals):
        ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
        
        # Add grid lines
        for i in range(degree_bounds[0] + 1):
            for j in range(degree_bounds[1] + 1):
                ax.plot([i, i], [j, j], [0, degree_bounds[2]], 
                       'gray', alpha=0.1)  # Vertical lines
        
        for i in range(degree_bounds[0] + 1):
            for k in range(degree_bounds[2] + 1):
                ax.plot([i, i], [0, degree_bounds[1]], [k, k], 
                       'gray', alpha=0.1)  # Y-direction lines
        
        for j in range(degree_bounds[1] + 1):
            for k in range(degree_bounds[2] + 1):
                ax.plot([0, degree_bounds[0]], [j, j], [k, k], 
                       'gray', alpha=0.1)  # X-direction lines
        
        # Plot all possible grid points
        points = [(i,j,k) for i in range(degree_bounds[0] + 1) 
                         for j in range(degree_bounds[1] + 1)
                         for k in range(degree_bounds[2] + 1)]
        if points:
            x_coords, y_coords, z_coords = zip(*points)
            ax.scatter(x_coords, y_coords, z_coords, c='lightgray', 
                      s=20, alpha=0.3, zorder=1)
        
        # Plot order ideal points
        order_ideal_x, order_ideal_y, order_ideal_z = zip(*order_ideal)
        
        if track_changes and idx > 0:
            # Split points into existing and new
            prev_points = set(order_ideals[idx-1])
            current_points = set(order_ideal)
            new_points = current_points - prev_points
            
            # Plot existing points in blue
            existing_points = [(x, y, z) for x, y, z in order_ideal 
                             if (x, y, z) not in new_points]
            if existing_points:
                ex_x, ex_y, ex_z = zip(*existing_points)
                ax.scatter(ex_x, ex_y, ex_z, c=base_color, s=60, alpha=0.7,
                          zorder=2, label='Existing points')
            
            # Plot new points in red
            if new_points:
                new_x, new_y, new_z = zip(*new_points)
                ax.scatter(new_x, new_y, new_z, c=new_point_color, s=60,
                          alpha=0.7, zorder=3, label='New points')
                ax.legend(loc='upper right', fontsize='small')
        else:
            # Plot all points in blue
            ax.scatter(order_ideal_x, order_ideal_y, order_ideal_z,
                      c=base_color, s=60, alpha=0.7, zorder=2)
        
        # Set labels and ticks
        ax.zaxis.set_rotate_label(False)
        ax.set_xlabel('deg(x)')
        ax.set_ylabel('deg(y)')
        ax.set_zlabel('deg(z)', rotation=0)
        
        # Set tick marks
        xticks = range(0, degree_bounds[0] + 1, max(1, degree_bounds[0] // 4))
        yticks = range(1, degree_bounds[1] + 1, max(1, degree_bounds[1] // 4))
        zticks = range(1, degree_bounds[2] + 1, max(1, degree_bounds[2] // 4))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks)
        
        ax.set_title(f'Step {idx+1}')
        ax.set_box_aspect([1,1,1])
        
        # Remove grid and background
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Set view angle
        ax.view_init(elev=20, azim=20)
        
        # Set axis limits
        ax.set_xlim(0, degree_bounds[0])
        ax.set_ylim(0, degree_bounds[1])
        ax.set_zlim(0, degree_bounds[2])
        
        # Fix axis juggling
        ax.xaxis._axinfo['juggled'] = (0,0,0)
        ax.yaxis._axinfo['juggled'] = (1,1,1)
        ax.zaxis._axinfo['juggled'] = (2,2,2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


# # Usage example
# if __name__ == "__main__":
#     # 2D example
#     bounds_2d = [3, 3]
#     order_ideal_2d = [(1,1), (2,1), (1,2)]
#     plot_order_ideal(bounds_2d, order_ideal_2d)
    
#     # 3D example
#     bounds_3d = [3, 3, 3]
#     order_ideal_3d = [(1,1,1), (2,1,1), (1,2,1), (1,1,2)]
#     plot_order_ideal(bounds_3d, order_ideal_3d)
