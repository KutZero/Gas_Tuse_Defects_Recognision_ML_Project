import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

IMAGE_SHAPE = (115,400,64)
ITEM_SIZE = 4 # bytes

def calc_mem_usage(input_image_shape, crop_size_arr, crop_step_arr, item_size):
    Z = np.zeros(crop_step_arr.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # расширение по всем сторонам
            #print(f'Crop size: {crop_size_arr[i,j]}, Crop step: {crop_step_arr[i,j]}')
            #print(f'Orig image size: {input_image_shape[0]}*{input_image_shape[1]}')
            rows_count = input_image_shape[0] + (crop_size_arr[i,j]-1)*2
            cols_count = input_image_shape[1] + (crop_size_arr[i,j]-1)*2
            #print(f'1 extended image size: {rows_count}*{cols_count}')
            
            # расширение для целого деления на кропы
            new_rows = crop_step_arr[i,j] - ((rows_count - crop_size_arr[i,j]) % crop_step_arr[i,j])
            new_cols = crop_step_arr[i,j] - ((cols_count - crop_size_arr[i,j]) % crop_step_arr[i,j])
            if new_rows != crop_step_arr[i,j]:
                rows_count += new_rows
            if new_cols != crop_step_arr[i,j]:
                 cols_count += new_cols
            #print(f'2 extended image size: {rows_count}*{cols_count}')
            
            # расчет кол-ва кропов
            crops_per_x_axis = len(np.zeros((rows_count-(crop_size_arr[i,j]-1)))[::crop_step_arr[i,j]])
            crops_per_y_axis = len(np.zeros((cols_count-(crop_size_arr[i,j]-1)))[::crop_step_arr[i,j]])
            crops_count = crops_per_x_axis * crops_per_y_axis
            
            # расчет кол-ва занимаемой памяти
            mem_usage = crops_count * crop_size_arr[i,j]**2 #* input_image_shape[2] * item_size
            #print(f'Crops count {crops_count}')
            #print(np.zeros((rows_count,cols_count)),end='\n\n')
            Z[i,j] = mem_usage
            
    return Z

crop_size_array = np.arange(1, IMAGE_SHAPE[0]+1, 1, dtype='int')
crop_step_array = np.arange(1, IMAGE_SHAPE[0]+1, 1, dtype='int') 
crop_size_array, crop_step_array = np.meshgrid(crop_size_array, crop_step_array)

mem_usage_array = calc_mem_usage(IMAGE_SHAPE, crop_size_array, crop_step_array, ITEM_SIZE)

plt.style.use('_mpl-gallery')

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

fig.set_figwidth(18)
fig.set_figheight(6)

ax.plot_surface(crop_size_array,
                crop_step_array,
                mem_usage_array, vmin = mem_usage_array.min() * 2, cmap=cm.Blues)

ax.set_xlabel('Размер кропа', fontsize=15) 
ax.set_ylabel('Шаг нарезки', fontsize=15) 
ax.set_zlabel('Затраты памяти', fontsize=15) 
ax.set_title(f'Зависимость затрат памяти от размера кропа и шага нарезки', fontsize=15) 

plt.show()