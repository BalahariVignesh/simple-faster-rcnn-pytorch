import cv2
import numpy as np


def read_labels(f):
    """Read the label data from file f and return a list of dicts of label parameters for 
        each object in the image/label file."""
    data = np.loadtxt(f, dtype=np.str, ndmin=2, delimiter=' ')
    
    def extract_row(row):
        result = {
            "type": row[0],  # 'Car', 'Pedestrian', ...
            "truncation": float(row[1]),  # truncated pixel ratio ([0..1])
            "occlusion": int(row[2]),  # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
            "alpha": float(row[3]),  # object observation angle ([-pi..pi])
            
            "x1": float(row[4]),  # left
            "y1": float(row[5]),  # top
            "x2": float(row[6]),  # right
            "y2": float(row[7]),  # bottom
        }

        # extract 3D bounding box information
        if len(row) > 8:
            result = {
                **result,
                "h": float(row[8]),  # box width
                "w": float(row[9]),  # box height
                "l": float(row[10]),  # box length
                "t(1)": float(row[11]),  # location (x)
                "t(2)": float(row[12]),  # location (y)
                "t(3)": float(row[13]),  # location (z)
                "ry": float(row[14])  # yaw angle
            }

        return result
    
    return list(map(extract_row, data))


def show_2d_bounding_box_image(img_file, label_file, filter_labels=None, line_width=2):
    """Given an image and corresponding label file, plot the 2d bounding boxes on the image and return result as CV image."""
    labels = read_labels(label_file)
    
    colors = {
        'Car': (228,26,28), 
        'Cyclist': (55,126,184), 
        'Misc': (77,175,74), 
        'Pedestrian': (152,78,163), 
        'Person_sitting': (255,127,0), 
        'Tram': (255,255,51), 
        'Truck': (166,86,40), 
        'Van': (247,129,191), 
        'DontCare': (153,153,153)
    }

    img = cv2.imread(img_file)

    for l in labels:
        if filter_labels is not None and l['type'] in filter_labels:
            cv2.rectangle(img, (int(l['x1']), int(l['y1'])), (int(l['x2']), int(l['y2'])), colors[l['type']], line_width)

    return img


def readCalibration(calib_dir, img_idx, cam):    
    # load 3x4 projection matrix
    f = '%s/%06d.txt' % (calib_dir, img_idx)
    with open(f,"r") as f:
        P = [x.split() for x in f.readlines()]
    
    P = np.array(P[cam])[1:]
    P = np.reshape(P, (4, 3)).T
    return P