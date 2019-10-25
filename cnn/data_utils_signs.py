import os
import sys
import pickle as pickle
import numpy as np
import json
import tensorflow as tf

from cnn.data_generator import SignDataLoader

classes = {
    "Rectangular": {
        "signs_classes": ["W13-1P_10", "W13-1P_15", "W13-1P_20", "W13-1P_25", "W13-1P_30",
                          "W13-1P_35", "W13-1P_45", "W16-7P", "W1-8_L", "W1-8_R", "W1-7",
                          "W1-6_L", "W1-6_R", "rectangle-other"],
        "h_symmetry": [("W1-6_L", "W1-6_R"), ("W1-8_L", "W1-8_R")],
        "rotation_and_flips": {"W1-7": ('h', 'v'),
                               "rectangle-other": ('v', 'h', 'd')}
    },
    "Diamond": {
        "signs_classes": ["W11-2", "W11-8", "W1-1_L", "W1-1_R", "W1-2_L", "W1-2_R", "W1-3_L", "W1-3_R", "W1-4_L",
                          "W1-4_R",
                          "W1-5_L", "W1-5_R", "W2-1", "W2-2_L", "W2-2_R", "W3-1", "W3-3", "W4-1_L", "W4-1_R", "W4-2",
                          "W5-2",
                          "W6-2", "W6-3", "W7-1", "W12-1", "W14-1", "W14-2", "diamond-other", "WorkZone"],
        # removed from training: "W1-1a_15_L"
        "h_symmetry": [("W1-1_L", "W1-1_R"), ("W1-2_L", "W1-2_R"), ("W1-3_L", "W1-3_R"), ("W1-4_L", "W1-4_R"),
                       ("W1-5_L", "W1-5_R"), ("W2-2_L", "W2-2_R"), ("W4-1_L", "W4-1_R"), ("W1-10_R", "W1-10_L")],
        "rotation_and_flips": {"W12-1": ('h',),
                               "W2-1": ('v', 'h', 'd'),
                               "W2-2_L": ('v',),
                               "W2-2_R": ('v',),
                               "W3-1": ('h',),
                               "W3-3": ('h',),
                               "W6-3": ('h',),
                               }
    },
    "Zebra": {
        "signs_classes": ["OM3-L", "OM3-R"],
        "h_symmetry": [("OM3-L", "OM3-R")],
        "rotation_and_flips": {"OM3-L": ('d',), "OM3-R": ('d',)}
    },

    "RedRoundSign": {
        "signs_classes": ['p1', 'p10', 'p11', 'p12', 'p19', 'p20', 'p23', 'p26', 'p27', 'p3', 'p5L', 'p6', 'p9', 'pax',
                          'pb', 'phx', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70',
                          'pl80', 'pmx', 'p_prohibited_bicycle_and_pedestrian', 'p_prohibited_bus_and_truck',
                          'p_prohibited_other', 'prx', 'p_other', 'plo'],
        "merge_sign_classes": {
            "prx": ['pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'prx'],
            "pmx": ['pm1.5', 'pm10', 'pm13', 'pm15', 'pm2', 'pm2.5', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46',
                    'pm5', 'pm49', 'pm50', 'pm55', 'pm8'],
            "phx": ['ph', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3.x', 'ph3',
                    'ph3.2', 'ph3.3', 'ph3.5', 'ph3.7', 'ph3.8', 'ph38', 'ph39', 'ph45', 'ph4', 'ph4.2', 'ph4.3',
                    'ph4.4', 'ph4.5', 'ph4.6', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'ph6'],
            "pax": ['pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pax'],
            "plo": ['pl35', 'pl25', 'pl15', 'pl10', 'pl110', 'pl65', 'pl90'],
            "p_other": ['pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5',
                        'p_prohibited_two_wheels_vehicules', 'p_prohibited_bicycle_and_pedestria',
                        'p_prohibited_bicycle_and_pedestrian_issues', 'p13', 'p15', 'p16', 'p17', 'p18', 'p2', 'p21',
                        'p22', 'p24', 'p25', 'p28', 'p4', 'p5R', 'p7L', 'p7R', 'p8', 'p15', 'pc']
            },
        "h_symmetry": [],
        "rotation_and_flips": {  # "pne": ('v', 'h', 'd'),
            # "pn": ('v', 'h', 'd'),
            # "pnl": ('d',),
            # "pc": ('v', 'h', 'd'),
            "pb": ('v', 'h', 'd'),
            "p_other": ('v', 'h', 'd'),
        }
    },
}


def get_data_for_master_class(class_name: str, mapping, mapping_id_to_name, rotation_and_flips, data_dir: str,
                              merge_sign_classes, h_symmetry_classes, image_size, ignore_npz: bool, out_classes,
                              test_to_train_ratio=0.0):
    data_file_path = "{0}/{0}.npz".format(class_name)
    if os.path.isfile(data_file_path) and not ignore_npz:
        savez = np.load(data_file_path)
        x_train = savez["x_train"]
        y_train = savez["y_train"]
        x_test = savez["x_test"]
        y_test = savez["y_test"]
    else:
        data_loader = SignDataLoader(path_images_dir=data_dir,
                                     classes_to_detect=out_classes,
                                     images_size=image_size,
                                     mapping=mapping,
                                     classes_flip_and_rotation=rotation_and_flips,
                                     symmetric_classes=h_symmetry_classes,
                                     train_test_split=test_to_train_ratio,
                                     classes_merge=merge_sign_classes)
        (x_train, y_train), (x_test, y_test) = data_loader.load_data()
        with open("{0}/{0}_class_counts.json".format(class_name), 'w') as count_json:
            train_names, train_counts = np.unique(y_train, return_counts=True)
            test_names, test_counts = np.unique(y_test, return_counts=True)
            counts = {n: {"train": 0, "test": 0} for n in mapping.keys()}
            for c, count in zip(train_names, train_counts):
                c_name = mapping_id_to_name[c]
                counts[c_name]["train"] = int(count)
            for c, count in zip(test_names, test_counts):
                c_name = mapping_id_to_name[c]
                counts[c_name]["test"] = int(count)
            json.dump(obj=counts, fp=count_json, indent=4)
        y_train = tf.keras.utils.to_categorical(y_train, len(out_classes))
        y_test = tf.keras.utils.to_categorical(y_test, len(out_classes))
        np.savez_compressed(data_file_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                            out_classes=out_classes)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    with open("{0}/{0}_mapping.json".format(class_name), 'w') as json_mapping:
        json.dump(mapping, json_mapping, indent=4)

    return x_train, y_train, x_test, y_test


def read_data(data_path, train_portion=1, input_size=(32, 32), ignore_npz=False, class_name="Diamond"):
    print("-" * 80)
    print("Reading data")

    images, labels = {}, {}

    out_classes = classes[class_name]["signs_classes"]
    rotation_and_flips = classes[class_name]["rotation_and_flips"]
    h_symmetry_classes = classes[class_name]["h_symmetry"]
    try:
        merge_sign_classes = classes[class_name]["merge_sign_classes"]
    except KeyError:
        merge_sign_classes = None

    mapping = {c: i for i, c in enumerate(out_classes)}
    mapping_id_to_name = {i: c for c, i in mapping.items()}

    os.makedirs(class_name, exist_ok=True)

    data = get_data_for_master_class(class_name=class_name,
                                     mapping=mapping,
                                     mapping_id_to_name=mapping_id_to_name,
                                     rotation_and_flips=rotation_and_flips,
                                     data_dir=data_path,
                                     merge_sign_classes=merge_sign_classes,
                                     h_symmetry_classes=h_symmetry_classes,
                                     image_size=input_size,
                                     ignore_npz=ignore_npz,
                                     out_classes=out_classes,
                                     test_to_train_ratio=1.0 - train_portion)
    images["train"], labels["train"], images["valid"], labels["valid"] = data

    print(images["train"].shape, labels["train"].shape, images["valid"].shape, labels["valid"].shape)

    images["test"], labels["test"] = None, None

    print("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std

    images["valid"] = (images["valid"] - mean) / std
    # images["test"] = (images["test"] - mean) / std

    return images, labels