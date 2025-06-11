import numpy as np
def whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):

     for j in index:
         den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
         data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den

     return data_f_whiten