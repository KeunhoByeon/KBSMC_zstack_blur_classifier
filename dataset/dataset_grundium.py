GRUNDIUM_SLIDE_INFO = {
    # Grundium_Z_layers_3_layers_2um
    "S23_01546_ER_GR_3layer": {"offset": 2, "stain": "immuno", "base_layer": 0},
    "S23_15652_Ecad_GR_3layer": {"offset": 2, "stain": "immuno", "base_layer": 0},
    "S23_17452_A_GR_3layer": {"offset": 2, "stain": "he", "base_layer": 0},
    "S23_17482_B_GR_3layer": {"offset": 2, "stain": "he", "base_layer": 0},

    # Grundium_Z_layers_5_layers_1um
    "MP2023004157_GR": {"offset": 1, "stain": "sish", "base_layer": 0},
    "S 2023016996_GR": {"offset": 1, "stain": "he", "base_layer": 0},
    "S 2023017057_GR": {"offset": 1, "stain": "he", "base_layer": 0},
    "S 2023017599_GR": {"offset": 1, "stain": "he", "base_layer": 0},
    "SE2023058353_GR": {"offset": 1, "stain": "he", "base_layer": 0},
    "SE2023058354_GR": {"offset": 1, "stain": "he", "base_layer": 0},
    "SE2023058399_GR": {"offset": 1, "stain": "he", "base_layer": 0},

    # z_Grundium_Z_Layer_test_adjust_offset_for_Blur_related
    "Grundium_EGFR_z_stack_offset5_5L_1um": {"offset": 1, "stain": "immuno", "base_layer": -2},
    "Grundium_EGFR_z_stack_offset5_5L_2um": {"offset": 2, "stain": "immuno", "base_layer": -2},
    "Grundium_EGFR_z_stack_offset5_5L_3um": {"offset": 3, "stain": "immuno", "base_layer": -2},
    "S24_00448_E_HE_5L_offset5_1um": {"offset": 1, "stain": "he", "base_layer": -2},
    "S24_00448_E_HE_5L_offset5_2um": {"offset": 2, "stain": "he", "base_layer": -2},
    "S24_00448_E_HE_5L_offset5_3um": {"offset": 3, "stain": "he", "base_layer": -2},

    # z_Grundium_Z_Layer_test_legacy_for_Blur_related
    "SE24_01714_1A_GR_5L_2um": {"offset": 2, "stain": "he", "base_layer": 0},
    "SE24_01714_1A_GR_5L_3um": {"offset": 3, "stain": "he", "base_layer": 0},
    "SE24_01714_1A_GR_5L_4um": {"offset": 4, "stain": "he", "base_layer": 0},
    "SE24_01714_1A_GR_5L_5um": {"offset": 5, "stain": "he", "base_layer": 0},
    "SE24_01714_Desmin_GR_5L_2um": {"offset": 2, "stain": "immuno", "base_layer": 0},
    "SE24_01714_Desmin_GR_5L_3um": {"offset": 3, "stain": "immuno", "base_layer": 0},
    "SE24_01714_Desmin_GR_5L_4um": {"offset": 4, "stain": "immuno", "base_layer": 0},
    "SE24_01714_Desmin_GR_5L_5um": {"offset": 5, "stain": "immuno", "base_layer": 0},
}

GRUNDIUM_SPLIT = {
    "train": [],
    "val": ["S24_00448_E_HE_5L_offset5_2um", "Grundium_EGFR_z_stack_offset5_5L_2um"],
    "test": ["SE24_01714_1A_GR_5L_2um", "SE24_01714_Desmin_GR_5L_2um",
             "SE24_01714_1A_GR_5L_3um", "SE24_01714_Desmin_GR_5L_3um"],
}
for slide_name in GRUNDIUM_SLIDE_INFO.keys():
    if slide_name not in GRUNDIUM_SPLIT["val"] and slide_name not in GRUNDIUM_SPLIT["test"]:
        GRUNDIUM_SPLIT["train"].append(slide_name)
