
# 'fix':['x','triangular_matrix','quarry_open_pit','crossed_downward_arrows','button','triangle','dot','c_dot','solid_colored_circle','asterix','purple_arrow_kite','diamond_words']
# 'rotate':['sleeping_y','small_inclined_fault_num','plus','reverse_p_num','barbeque_tofu','fault_line_triangle_num','line_diamond_center_solid','small_inclined_fault','inclined_fault',
# 'small_inclined_triangle_fault_num','christmas_tree','fault_line_triangle_hollow_num','line_diamond_center','arrow_circle','arrow_num',]

# python3 train_points_64p.py --label_key_name='sleeping_y'  --center_crop_ratio=0.4

# python3 test_dnn.py --key='x' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_good'
# python3 test_dnn.py --key='triangle' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_good'
# python3 test_dnn.py --key='triangular_matrix' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_good'
# python3 test_dnn.py --key='button' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_good'
# python3 test_dnn.py --key='crossed_downward_arrows' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_good'
# python3 test_dnn.py --key='dot' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_good'
# python3 test_dnn.py --key='barbeque_tofu' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_good'

# python3 test_dnn.py --key='reverse_p_num' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
# python3 test_dnn.py --key='small_inclined_fault_num' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
# python3 test_dnn.py --key='plus' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
# python3 test_dnn.py --key='quarry_open_pit' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'
# python3 test_dnn.py --key='sleeping_y' --checkpoint_dir='/data2/mineral_competition/zekun_models/1109/' --output_dir='/data2/mineral_competition/zekun_outputs_1109_low'

python3 train_points.py --label_key_name='sleeping_y' --rot_aug --model_size='large' --batch_size=32 # 224

python3 train_points.py --label_key_name='fault_line_triangle_num' --rot_aug --model_size='small' --batch_size=32  # 185
python3 train_points.py --label_key_name='line_diamond_center_solid' --rot_aug --model_size='small' --batch_size=32  # 194
python3 train_points.py --label_key_name='small_inclined_fault_num' --rot_aug --model_size='large' --batch_size=16 #238
python3 train_points.py --label_key_name='quarry_open_pit' --model_size='large' --batch_size=32 #195

python3 train_points.py  --label_key_name='asterix' --model_size='large' --batch_size=32 #238
python3 train_points.py  --label_key_name='purple_arrow_kite' --model_size='large' --batch_size=32 #185
python3 train_points.py  --label_key_name='line_diamond_center_solid' --model_size='large' --batch_size=32 --rot_aug # 194



python3 test_dnn.py --key='quarry_open_pit' --checkpoint_dir='/data2/mineral_competition/zekun_models/checkpoints/' --output_dir='/data2/mineral_competition/zekun_outputs_1111'
python3 test_dnn.py --key='sleeping_y' --checkpoint_dir='/data2/mineral_competition/zekun_models/checkpoints/' --output_dir='/data2/mineral_competition/zekun_outputs_1111'


# final test
python3 test_dnn.py --key='x' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='triangular_matrix' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='triangle' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='button' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='crossed_downward_arrows' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'

python3 test_dnn.py --key='dot' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='quarry_open_pit' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='sleeping_y' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='purple_arrow_kite' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'
python3 test_dnn.py --key='line_diamond_center_solid' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/dnn'

python3 test_dnn.py --key='small_inclined_fault_num' --checkpoint_dir='/data2/mineral_competition/zekun_models/test/' --output_dir='/data2/mineral_competition/zekun_test/add'


# python3 train_points.py --label_key_name='small_inclined_fault_num' --rot_aug --model_size='large' --batch_size=16 #238
