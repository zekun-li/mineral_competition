

# python3 train_points_64p.py --label_key_name='x'
# python3 train_points_64p.py --label_key_name='sleeping_y'  
# python3 train_points_64p.py --label_key_name='triangular_matrix'
# python3 train_points_64p.py --label_key_name='quarry_open_pit'
# python3 train_points_64p.py --label_key_name='crossed_downward_arrows'

# python3 train_points_64p.py --label_key_name='small_inclined_fault_num'
# python3 train_points_64p.py --label_key_name='button'
# python3 train_points_64p.py --label_key_name='plus'
# python3 train_points_64p.py --label_key_name='reverse_p_num'
# python3 train_points_64p.py --label_key_name='triangle'
# python3 train_points_64p.py --label_key_name='barbeque_tofu'
# python3 train_points_64p.py --label_key_name='fault_line_triangle_num'

# python3 train_points_64p.py --label_key_name='line_diamond_center_solid'
# python3 train_points_64p.py --label_key_name='dot'
# python3 train_points_64p.py --label_key_name='c_dot'
# python3 train_points_64p.py --label_key_name='solid_colored_circle'
# python3 train_points_64p.py --label_key_name='asterix'
# python3 train_points_64p.py --label_key_name='small_inclined_fault'
# python3 train_points_64p.py --label_key_name='inclined_fault'
# python3 train_points_64p.py --label_key_name='small_inclined_triangle_fault_num'
# python3 train_points_64p.py --label_key_name='purple_arrow_kite'
# python3 train_points_64p.py --label_key_name='christmas_tree'
# python3 train_points_64p.py --label_key_name='fault_line_triangle_hollow_num'
# python3 train_points_64p.py --label_key_name='line_diamond_center'
# python3 train_points_64p.py --label_key_name='diamond_words'


# python3 train_points.py --label_key_name='x'
# python3 train_points.py --label_key_name='sleeping_y'
# python3 train_points.py --label_key_name='triangular_matrix'
# python3 train_points.py --label_key_name='quarry_open_pit'
# python3 train_points.py --label_key_name='crossed_downward_arrows'


# 'fix':['x','triangular_matrix','quarry_open_pit','crossed_downward_arrows','button','triangle','dot','c_dot','solid_colored_circle','asterix','purple_arrow_kite','diamond_words']
# 'rotate':['sleeping_y','small_inclined_fault_num','plus','reverse_p_num','barbeque_tofu','fault_line_triangle_num','line_diamond_center_solid','small_inclined_fault','inclined_fault',
# 'small_inclined_triangle_fault_num','christmas_tree','fault_line_triangle_hollow_num','line_diamond_center','arrow_circle','arrow_num',]

# python3 train_points_64p.py --label_key_name='sleeping_y'  --center_crop_ratio=0.4

python3 test_dnn.py --key='x' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109'
python3 test_dnn.py --key='triangle' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109'
python3 test_dnn.py --key='triangular_matrix' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109'
python3 test_dnn.py --key='button' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109'
python3 test_dnn.py --key='crossed_downward_arrows' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109'
python3 test_dnn.py --key='dot' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109'
python3 test_dnn.py --key='barbeque_tofu' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109'

python3 test_dnn.py --key='reverse_p_num' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
python3 test_dnn.py --key='small_inclined_fault_num' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
python3 test_dnn.py --key='plus' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
python3 test_dnn.py --key='quarry_open_pit' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
python3 test_dnn.py --key='sleeping_y' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'


# python3 test_dnn.py --key='quarry_open_pit'
# python3 test_dnn.py --key='dot'