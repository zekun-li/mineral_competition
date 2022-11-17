index_list_dict = {
      'x' : [1, 6, 12, 17, 19, 54], 
      'triangular_matrix': [0, 5, 11, 16, 18, 53],     
      'triangle' : [62], 
      'button': [ 29],
      'crossed_downward_arrows' : [4,10,15,], 
      'dot':[9, 21, 42,55, 56, 59, 61, 63], 
      'barbeque_tofu':[],

      'quarry_open_pit' : [3, 14],
      'sleeping_y':[2,7,13, 20,], # --------

      'reverse_p_num':[49],
      'small_inclined_fault_num':[ 28, 48 ],
      'plus':[50], # no so good
      'line_diamond_center_solid':[46],
      'fault_line_triangle_num':[44],
      'purple_arrow_kite':[22],
      'asterix':[23]
}

# template:  23, 24, 30,31, 43, 45,47, 51, 52, 57, 58, 59, 60, 

color_indices = [22, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
                ]
# # change 213 to actual test set size
# template_matching_indices = list(set([i for i in range(0, XXXXX)]) - set([item for sublist in list(index_list_dict.values()) for item in sublist] + color_indices))

template_matching_indices = sorted(list(set([i for i in range(0, 64)]) - set([item for sublist in list(index_list_dict.values()) for item in sublist] + color_indices))) 
