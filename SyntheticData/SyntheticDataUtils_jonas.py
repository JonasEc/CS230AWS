import pandas as pd
import numpy as np

import cv2
import mnist


def load_mnist_data(path=None):
    if path is None:
        train_images_array = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
        train_labels_array = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    else:
        train_images_array = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz", path)
        train_labels_array = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz", path)
        
    image_dict_array = [ 
        { 
            "image" : train_images_array[i],
            "label" : train_labels_array[i]
        } for i in range(len(train_images_array))
    ]
    return image_dict_array


def get_random(image_dict_array):
    random_index = int(np.floor(len(image_dict_array) * np.random.rand()))
    use_label = image_dict_array[random_index]["label"]
    use_image = image_dict_array[random_index]["image"]
    return use_label, use_image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def get_num_zeros_by_row(image):
    num_rows = image.shape[0]
    results_array = []
    for num_row, row in enumerate( image ):
        rightmost_nonzero_index = max([ i if row[i] > 0 else 0 for i in range(len(row))])
        leftmost_nonzero_index = min([ i if row[i] > 0 else len(row) for i in range(len(row))])
        
        if rightmost_nonzero_index > 0:
            rightmost_nonzero_index += 1
        num_right_zeros = len(row) - rightmost_nonzero_index 
        
        num_left_zeros = leftmost_nonzero_index

        results_array.append({"row" : num_row, "left" : num_left_zeros, "right" : num_right_zeros})
    results_df = pd.DataFrame(results_array)
    results_df.index = results_df["row"]
    results_df.sort_index(ascending=True, inplace=True)
    return results_df[["left", "right"]]

def random_comma(h,w):
    img = np.zeros([h,w])
    rVert  = np.random.randint(np.floor( h*0.75-6 ),h-6,1)[0]
    rHoriz = np.random.randint(0,w-3,1)[0]
    img[rVert:(rVert+6), rHoriz:(rHoriz+3)] = 255
    img = rotate_image(img, np.random.randint(-45,45,1)[0])
    return img

def concat_images_remove_gap(left_image, right_image, gap_tolerance = 0):
    left_zeros_df = get_num_zeros_by_row(left_image)
    right_zeros_df = get_num_zeros_by_row(right_image)
    join_zeros_df = pd.concat( [ left_zeros_df[["right"]], right_zeros_df[["left"]]], axis=1)
    join_zeros_df["gap"] = join_zeros_df["right"] + join_zeros_df["left"]
    min_gap = min(join_zeros_df["gap"])
    #print("Min gap = ", min_gap)

    zeros_to_remove = min_gap - gap_tolerance

    #print("Zeros to remove = ", zeros_to_remove)
    
    rows_array = []
    for num_row in range(len(left_image)):
        left_image_right_side_zeros = int(join_zeros_df.loc[num_row]["right"])
        right_image_left_side_zeros = int(join_zeros_df.loc[num_row]["left"])
        total_num_zeros = left_image_right_side_zeros + right_image_left_side_zeros
        try:
            left_image_row_no_zeros = left_image[num_row][:-left_image_right_side_zeros]
        except:
            raise Exception(left_image_right_side_zeros, num_row, left_image[num_row])
        assert( len(left_image_row_no_zeros) + left_image_right_side_zeros == len(left_image[num_row]))
        
        right_image_row_no_zeros = right_image[num_row][right_image_left_side_zeros:]
        assert( len(right_image_row_no_zeros) + right_image_left_side_zeros == len(right_image[num_row]))
        
        assert len(left_image_row_no_zeros) + len(right_image_row_no_zeros) + total_num_zeros == len(left_image[num_row]) + len(right_image[num_row])

        num_zeros_to_add = total_num_zeros - zeros_to_remove
        concatenated_row = np.concatenate( [ left_image_row_no_zeros, np.zeros([num_zeros_to_add]), right_image_row_no_zeros ] )
        rows_array.append(concatenated_row)

    output_image = np.array(rows_array)
    return output_image


def random_resize(image, target_width, target_height):
    resizeHeight = np.random.randint(image.shape[0], target_height, 1)[0]
    resizeWidth  = np.random.randint(image.shape[1], target_width,  1)[0]
    resize = cv2.resize(image, dsize= (resizeWidth,resizeHeight))
    return resize


def randomly_pad_size(image, target_width, target_height):
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    assert target_width >= image_width, "Target width is less than image width!"
    assert target_height >= image_height, "Target width is less than image width!"
    diff_width = target_width - image_width
    diff_height = target_height - image_height

    left_num_zeros = int(np.floor(np.random.rand() * diff_width))
    right_num_zeros = diff_width - left_num_zeros
    top_num_zeros = int(np.floor(np.random.rand() * diff_height))
    bottom_num_zeros = diff_height - top_num_zeros

    new_image = np.concatenate([ np.zeros([image.shape[0], left_num_zeros]), 
                                 image, 
                                 np.zeros([image.shape[0], right_num_zeros])] , axis=1)
    new_image = np.concatenate([ np.zeros([top_num_zeros, new_image.shape[1]]),
                                 new_image, 
                                 np.zeros([bottom_num_zeros, new_image.shape[1]])] , axis=0)  
    return new_image

def random_cut_and_pad(random_image, target_height, where):
    random = np.random.randint(0,random_image.shape[0]/2, 1)[0]
    if where == 'from_above':
        random_image = random_image[(random_image.shape[0] - random):]
        random_image = np.concatenate([random_image, np.zeros([target_height-random, random_image.shape[1]])] , axis=0)  
    elif where == 'from_below':
        random_image = random_image[:random]
        random_image = np.concatenate([np.zeros([target_height-random, random_image.shape[1]]), random_image] , axis=0)  
    else:
        random_iamge = np.zeros([target_height, random_image.shape[1]]) 
    return random_image


def add_noise(image, p_noise = 1e-2):
    [ height, width ] = image.shape
    for i in range(height):
        for j in range(width):
            if np.random.rand() < p_noise:
                pixel_value = image[i][j]
                if pixel_value == 0: # if black, add random greyscale
                    new_value = int(np.floor( np.random.rand() * 256 ))
                else: # if not black, randomly darken
                    new_value = int(np.floor( np.random.rand() * pixel_value ))
                image[i][j] = new_value
    return image


def concat_images(image_array):
    return np.concatenate(image_array, axis=1)


def add_lines(total_height,total_width):
    img = np.zeros((total_height,total_width))
    for k in range(4):
        yes = np.random.uniform()
        if yes > 0.5:
            r = np.random.randint(0,5,1)[0]
            if k == 0:
                img[r:r+3,:] = np.random.randint(125,255,[3,total_width])
            elif k == 1:
                img[(total_height-r-3):(total_height-r),:] = np.random.randint(125,255,[3,total_width])                
            elif k == 2:
                img[:, r:r+3] = np.random.randint(125,255,[total_height,3])
            elif k == 3:
                img[:,(total_width-r-3):(total_width-r)] = np.random.randint(125,255,[total_height, 3])
    img = rotate_image(img, np.random.randint(-5,5,1)[0])
    return img


def create_digit_string(image_dict_array, config_dict):
    num_images = len(image_dict_array)

    total_width = config_dict.get("Total width")
    total_height = config_dict.get("Total height")
    max_num_digits = config_dict.get("Max num digits")
    max_rotation_degrees = config_dict.get("Max rotation degrees")
    p_gaussian_noise = config_dict.get("P Gaussian noise")

    num_digits = int(1 + np.floor(np.random.rand() * max_num_digits))
    #print(" ".join(["Using", str(num_digits), "digits"]))

    concat_image_array = []
    concat_digit_array = []

    comma =False
    for i in range(num_digits):
        # grab a random image
        use_label, use_image = get_random(image_dict_array)
        random_index = int(np.floor(num_images * np.random.rand()))

        use_label = image_dict_array[random_index]["label"]
        use_image = image_dict_array[random_index]["image"]

        # apply a random rotation
        use_angle = max_rotation_degrees * (np.random.rand() - 0.5)
        modified_image = rotate_image(use_image, use_angle)

        concat_image_array.append(modified_image)
        concat_digit_array.append(use_label)
        
        r = np.random.uniform()
        if r > 0.65 and comma == False and i < (num_digits-1):
            concat_image_array.append(random_comma(modified_image.shape[0], modified_image.shape[1]))
            concat_digit_array.append(11) 
            comma =True

    # join images laterally, removing gaps
    concatenated_image = concat_image_array[0]
    for right_image in concat_image_array[1:]:
        concatenated_image = concat_images_remove_gap(concatenated_image, right_image)

    # random resize to ensure we have different scale digits
    concatenated_image = random_resize(concatenated_image, total_width, total_height)
    # now pad the size
    concatenated_image = randomly_pad_size(concatenated_image, total_width, total_height)
    
    concatenated_image = add_noise(concatenated_image, p_gaussian_noise)
    concatenated_label = "_".join([str(x) for x in concat_digit_array])

    # now add a random numbers from above and below
    for k in range(8):
        __, random_image = get_random(image_dict_array)
        random_image = random_resize(random_image, total_width/2, random_image.shape[0] + 4)
        random_image = randomly_pad_size(random_image, concatenated_image.shape[1], random_image.shape[0])
        if k%2 == 0:
            which = 'from_below'
        elif k%2 == 1:
            which = 'from_above'        
        random_image = random_cut_and_pad(random_image, total_height, which)
        concatenated_image = concatenated_image + random_image

#     # now add a random number from below
#     __, random_image = get_random(image_dict_array)
#     random_image = random_resize(random_image, total_width/2, random_image.shape[0] + 4)
#     random_image = randomly_pad_size(random_image, concatenated_image.shape[1], random_image.shape[0])
#     random_image = random_cut_and_pad(random_image, total_height, 'from_below')
#     concatenated_image = concatenated_image + random_image
    
    concatenated_image = add_lines(total_height,total_width) + concatenated_image
    concatenated_image = rotate_image(concatenated_image, np.random.randint(-5,5,1)[0])
    
    concatenated_image = np.minimum(concatenated_image,255)
    
    return {"Label" : concatenated_label, "Image" : concatenated_image, "Metadata" : {}}

def makeDataSet(image_dict_array, config_dict, N):
    MAXCHARLENGTH = 6
    BATCH_SIZE = 32
    IMGWIDTH = 128
    IMGHEIGHT = 64
    
    imgs = []
    labels = []
    labelLength = []
    k = 0
    while k < N:
        try:
            result_dict = create_digit_string(image_dict_array, config_dict)
            imgs.append(result_dict['Image']/255)
            labels.append(np.array([int(k) for k in result_dict['Label'].split('_')]))
            labelLength.append(labels[-1].size)
            k +=1
        except:
            print(k)   
    imgs = np.array(imgs)
    labels = np.array(labels)
    labelLength = np.array(labelLength)
    
    Y_len = labelLength.reshape([N,1])
    X = imgs.reshape([N, IMGHEIGHT, IMGWIDTH, 1])
    
    Y_label = np.ones([N , MAXCHARLENGTH]) * -1
    for k in range(N):
        Y_label[k, :int(Y_len[k])] = labels[k]
              
    return X, Y_label, Y_len
                          
                          
                          