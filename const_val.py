index_list_dict = {
      'x' : [17,22,37,45,47,54,58,62,66,69,109,114,121,125,129,132,137,141,145,149,153,158,
          162,164,], 
      'triangle' : [28, 185,192], 
      'triangular_matrix': [21,36,44,53,57,61,65,68,75,108,113,124,128,131,136,140,
                          144,148,152,157,161,163,],            
      'button': [26,35, 48,119,168 ], # 35?
      'crossed_downward_arrows' : [20,56,112,123,135,147,151,156,204,207,209,211,], # new 112
      'dot':[9,18,31,174,184,191,], # new: 18
      'barbeque_tofu':[32],

      'reverse_p_num':[50,73],
      'small_inclined_fault_num':[19, 27, 30, 42, 49, 120,167, 169, ],
      'plus':[52], # no so good
      # 'line_diamond_center_solid':[43],

      'quarry_open_pit' : [24,39,60,64,107,111,116,122,127,134,139,143,155,160,166,171,172,
                        176,206,208,210,212],
      'sleeping_y':[23,38,46,55,59,63,67,70, 110,115,126,130,133,138,142,146, 150,154,159,
                 165,205,], # --------
      # 'x':[17],
      # 'triangular_matrix':[53],
      # 'crossed_downward_arrows':[112],
      # 'dot':[18]
    
}


color_indices = [0,1,2,3,4,5,10,25, 33,77,78,79,80,
                 81,82,83,84,85,86,87,88,89,90,91,92,93,94,
                 95,96,97,98,99,100,101,102,103,104,105,106,
                 118,179,180,181,182,183,186,187,188,189,190,
                ]
# # change 213 to actual test set size
# template_matching_indices = list(set([i for i in range(0, XXXXX)]) - set([item for sublist in list(index_list_dict.values()) for item in sublist] + color_indices))

template_matching_indices = sorted(list(set([i for i in range(0, 213)]) - set([item for sublist in list(index_list_dict.values()) for item in sublist] + color_indices) - set([51])))

legend_path_dict = {'button':['AZ_GrandCanyon_label_horiz_bedding_pt.jpeg', # difficult
                              'CO_Alamosa_label_Horizontal_bedding_pt.jpeg',
                              'NM_Volcanoes_label_horizon_bedding_pt.jpeg',
                              'OR_Carlton_label_bedding_horiz_pt.jpeg'],
                     
                     'triangle':['AZ_GrandCanyon_label_sinkhole_pt.jpeg',
                                'Trend_2007_fig10_1_label_sample_pt.jpeg',
                                'Trend_2007_fig10_2_label_sample_pt.jpeg'],
                     
                     'triangular_matrix': ['CA_Coulterville_297201_1947_62500_geo_mosaic_label_2_pt.jpeg'],
                     
                     'x':['CA_Coulterville_297201_1947_62500_geo_mosaic_label_3_pt.jpeg',
                         'AZ_Clifton_314492_1962_62500_geo_mosaic_label_3_pt.jpeg',
                         'NV_Cortez_320656_1938_48000_geo_mosaic_label_3_pt.jpeg',
                         'CO_Climax_400825_1970_24000_geo_mosaic_label_3_pt.jpeg'],
                     
                     'crossed_downward_arrow':['AZ_Clifton_314492_1962_62500_geo_mosaic_label_1_pt.jpeg',
                                              'NV_EdnaMountain_320880_1965_62500_geo_mosaic_label_1_pt.jpeg',
                                              'WI_Hatfield_503290_1958_48000_geo_mosaic_label_1_pt.jpeg',
                                              'WI_Waupaca_503601_1957_48000_geo_mosaic_label_1_pt.jpeg',
                                              'WI_Wausau_503603_1953_48000_geo_mosaic_label_1_pt.jpeg',
                                              'WI_Wautoma_503608_1959_48000_geo_mosaic_label_1_pt.jpeg'],
                     
                     'quarry_open_pit':['CA_Coulterville_297201_1947_62500_geo_mosaic_label_5_pt.jpeg',
                                        'AZ_Clifton_314492_1962_62500_geo_mosaic_label_5_pt.jpeg',
                                        'CO_Climax_400825_1970_24000_geo_mosaic_label_5_pt.jpeg',
                                        'CO_MountSherman_401586_1961_24000_geo_mosaic_label_5_pt.jpeg',
                                        'NV_Cortez_320656_1938_48000_geo_mosaic_label_5_pt.jpeg',
                                        'KY_Harlan_804185_1903_48000_geo_mosaic_label_5_pt.jpeg'], # difficult
                    'line_diamond_center_solid':['CA_Elsinore_label_vert_meta_foliation_pt.jpeg'],
                    'dot':['46_Coosa_2015_11 74_label_drill_pt.jpeg',
                          'AZ_PipeSpring_label_collapse_structure_pt.jpeg',
                          'Titac_2011_fig7_6_label_USS_drill_pt.jpeg',
                          'Trend_2007_fig10_2_label_drill_pt.jpeg'],
                    
                    }