from django.core.files.storage import default_storage
from django.conf import settings
from . import urls
from django.shortcuts import render
from django.http import HttpResponse
import requests
import os 
import random
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import cv2
import matplotlib.pyplot as plt

# global variables 
target_path="source.png"


def load_image(source_path,mask_path,target_path):
    image_data = {}
    source = cv2.imread(source_path) # source
    mask = cv2.imread(mask_path) # mask
    target = cv2.imread(target_path) # target

    # normalize the images
    image_data['source'] = cv2.normalize(source.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['mask'] = cv2.normalize(mask.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['target'] = cv2.normalize(target.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['dims'] = [96,96]

    return image_data

def load_image_flipped(source_path,mask_path,target_path):
    image_data = {}
    source = cv2.imread(source_path) # source
    mask = cv2.imread(mask_path) # mask
    target = cv2.imread(target_path) # target

    source=cv2.flip(source, 1)

    # normalize the images
    image_data['source'] = cv2.normalize(source.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['mask'] = cv2.normalize(mask.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['target'] = cv2.normalize(target.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['dims'] = [96,96]

    return image_data

def display_image(image_data):
    # show the image
    plt.figure(figsize=(16,16))
    for i in range(3):
        if(i == 0):
            img_string = 'source'
        elif(i == 1):
            img_string = 'mask'
        else:
            img_string = 'target'
        img = image_data[img_string]
        plt.subplot(1,3,i+1)
        plt.imshow(img[:,:,[2,1,0]])

def find_mask_patch_coordinates(mask_image, pixel_value):
    # Find all non-zero pixels in the mask image
    nonzero_pixels = np.argwhere(mask_image == pixel_value)

    # If no pixels with the specified value found, return None
    if len(nonzero_pixels) == 0:
        return None

    # Get the minimum and maximum row and column indices
    min_row = np.min(nonzero_pixels[:, 0])
    max_row = np.max(nonzero_pixels[:, 0]) + 1
    min_col = np.min(nonzero_pixels[:, 1])
    max_col = np.max(nonzero_pixels[:, 1]) + 1

    # Calculate the height and width of the patch region
    patch_height = max_row - min_row 
    patch_width = max_col - min_col 

    # Return the coordinates of the patch region
    return (min_row, max_row), (min_col, max_col), (patch_width,patch_height)

def reshape_image(image, desired_size):
    """
    Reshape the input image to the desired size.
    
    Parameters:
        image (numpy.ndarray): Input image.
        desired_size (tuple): Desired size in the format (width, height).
    
    Returns:
        numpy.ndarray: Reshaped image.
    """
    # Resize the image to the desired size
    reshaped_image = cv2.resize(image, desired_size, interpolation=cv2.INTER_NEAREST)
    return reshaped_image

def preprocess(image_data):
    # extract image data
    source = image_data['source']
    mask = image_data['mask']
    target = image_data['target']

    # get image shape and offset
    Hs,Ws,_ = source.shape
    Ht,Wt,_ = target.shape
    Ho, Wo = image_data['dims']

    # get coordinates of mask patch
    x,y,sp=find_mask_patch_coordinates(mask,1.)
    H_min,H_max=x
    W_min,W_max=y
    W_p,H_p=sp
    
    # check if source img shape matches the mask shape
    if (Hs,Ws)!=(H_p,W_p):
        source=reshape_image(source,(W_p,H_p))
    
    return {'source':source, 'mask': mask, 'target': target, 'dims':[H_min,H_max,W_min,W_max]}

# performs naive cut-paste from source to target
def naive_copy(image_data):
    # extract image data
    source = image_data['source']
    mask = image_data['mask']
    target = image_data['target']
    dims = image_data['dims']
    
    target[dims[0]:dims[1],dims[2]:dims[3],:] = target[dims[0]:dims[1],dims[2]:dims[3],:] * (1 - mask) + source * mask
    
    return target

def mask_index_to_patch_index(x, mask_width, patch_bounds, patch_width, patch_height):
    # Calculate the row and column of the flattened index x
    row = x // mask_width
    col = x % mask_width
    
    # Check if the index x falls within the bounds of the white patch
    if patch_bounds[0] <= row < patch_bounds[1] and patch_bounds[2] <= col < patch_bounds[3]:
        # Calculate the position of x relative to the top-left corner of the patch
        patch_row = row - patch_bounds[0]
        patch_col = col - patch_bounds[2]
        
        # Calculate the corresponding index y within the patch
        y = patch_row * patch_width + patch_col
        return y
    else:
        return 0

def get_subimg(image, dims):
    return image[dims[0]:dims[1], dims[2]:dims[3]]

def poisson_blending(image, GRAD_MIX):
    # comparison function
    def _compare(val1, val2):
        if(abs(val1) > abs(val2)):
            return val1
        else:
            return val2
  
    # membrane (region where Poisson blending is performed)
    mask = image['mask']
    Hs,Ws = mask.shape
    num_pxls = Hs * Ws
    H_p,W_p=image['source'].shape

    # source and target image
    source = image['source'].flatten(order='C')
    target_subimg = get_subimg(image['target'], image['dims']).flatten(order='C')
    target=image['target'].flatten(order='C')

    # initialise the mask, guidance vector field and laplacian
    mask = mask.flatten(order='C')
    guidance_field = np.empty_like(mask)
    laplacian = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')
    
    for i in range(num_pxls):
        # construct the sparse laplacian block matrix
        # and guidance field for the membrane
    
        if(mask[i] > 0.99):
            
            laplacian[i, i] = 4

            # get index inside mask patch    
            patch_index=mask_index_to_patch_index(i,Ws,image['dims'],W_p,H_p)

            # construct laplacian, and compute source and target gradient in mask
            if(i - Ws > 0 and patch_index - W_p + 1 >0):
                
                laplacian[i, i-Ws] = -1
            
                # calculate according to patch_index and previous_patch_index
                Np_up_s = source[patch_index] - source[patch_index-W_p]
                Np_up_t = target_subimg[patch_index] - target_subimg[patch_index-W_p]
                
            else:
                Np_up_s = source[patch_index]
                Np_up_t = target_subimg[patch_index]

            if(i % Ws != 0 and patch_index % W_p != 0):
               
                laplacian[i, i-1] = -1
                Np_left_s = source[patch_index] - source[patch_index-1]
                Np_left_t = target_subimg[patch_index] - target_subimg[patch_index-1]
                
            else:   
                Np_left_s = source[patch_index]
                Np_left_t = target_subimg[patch_index]

            if(i + Ws < num_pxls and patch_index + W_p < W_p*H_p):
                
                laplacian[i, i+Ws] = -1
                
                Np_down_s = source[patch_index] - source[patch_index + W_p]
                Np_down_t = target_subimg[patch_index] - target_subimg[patch_index + W_p]
            else:
                Np_down_s = source[patch_index]
                Np_down_t = target_subimg[patch_index]

            if(i % Ws != Ws-1 and patch_index % W_p != W_p-1):
                
                laplacian[i, i+1] = -1
                Np_right_s = source[patch_index] - source[patch_index+1]
                Np_right_t = target_subimg[patch_index] - target_subimg[patch_index+1]
                
            else:
                Np_right_s = source[patch_index]
                Np_right_t = target_subimg[patch_index]

            # choose stronger gradient
            if(GRAD_MIX is False):
                Np_up_t = 0
                Np_left_t = 0
                Np_down_t = 0
                Np_right_t = 0

            guidance_field[i] = (_compare(Np_up_s, Np_up_t) + _compare(Np_left_s, Np_left_t) + 
                                _compare(Np_down_s, Np_down_t) + _compare(Np_right_s, Np_right_t))

        else:
            # if point lies outside membrane, copy target function
            laplacian[i, i] = 1
            guidance_field[i] = target[i]
  
    return [laplacian, guidance_field]

# linear least squares solver
def linlsq_solver(A, b, dims):
    x = linalg.spsolve(A.tocsc(),b)
    return np.reshape(x,(dims[0],dims[1]))

# stitches poisson equation solution with target
def stitch_images(source, target, dims):
    target[dims[0]:dims[1], dims[2]:dims[3],:] = source[dims[0]:dims[1], dims[2]:dims[3],:]
    return target

# performs poisson blending
def blend_image(data, BLEND_TYPE, GRAD_MIX):
    if(BLEND_TYPE == 1):
        image_solution = naive_copy(data)
    
    elif(BLEND_TYPE == 2):
        equation_param = []
        ch_data = {}
    
        # construct poisson equation 
        for ch in range(3):
            ch_data['source'] = data['source'][:,:,ch]
            ch_data['mask'] = data['mask'][:,:,ch]
            ch_data['target'] = data['target'][:,:,ch]
            ch_data['dims'] = data['dims']
            equation_param.append(poisson_blending(ch_data, GRAD_MIX))

        # solve poisson equation
        image_solution = np.empty_like(data['target'])
        for i in range(3):
            image_solution[:,:,i] = linlsq_solver(equation_param[i][0],equation_param[i][1],data['target'].shape)
            print(type(image_solution),image_solution.shape)
        image_solution = stitch_images(image_solution,data['target'],data['dims'])
    
    else:
        # wrong option
        raise Exception('Wrong option! Available: 1. Naive, 2. Poisson')
      
    return image_solution


def get_random_items(lst,n):
    return random.sample(lst, n)

def homepage(request):
    return render (request, "index.html")

def services(request):
    return render (request, "services.html")

def about(request):
    return render (request, "about.html")

def contact(request):
    return render (request, "contact.html")

def eye(request):
    random_images=[os.path.join("static/eye_cluster",i) for i in os.listdir("static/eye_cluster")]
    #random_images = get_random_items(x,8)
    
    return render (request, "single.html",{'raw':random_images,'main':"EYE"})

def mouth(request):
    random_mouth=[os.path.join("static/mouth_clusters",i) for i in os.listdir("static/mouth_clusters")]
    #random_mouth = get_random_items(x,8)
    
    return render (request, "single.html",{'raw':random_mouth,'main':"Mouth"})

def nose(request):
    random_images=[os.path.join("static/nose_clusters",i) for i in os.listdir("static/nose_clusters")]
    #random_nose = get_random_items(x,12)
    
    return render (request, "single.html",{'raw':random_images, 'main':"Nose"})

def select_eye(request):
    random_images=[os.path.join("static/eye_cluster",i) for i in os.listdir("static/eye_cluster")]
    #random_images = get_random_items(x,8)
    
    return render (request, "single_form.html",{'raw':random_images,'main':"EYE", "next_action":"form2"})

def select_nose(request):
    if request.method == 'POST':
        selected_category = request.POST.get('image_category')
        source_path=selected_category
        BLEND_TYPE = 2
        GRAD_MIX = True
        mask_path= "Left_eye_mask.png"
        target_path="source.png"
        image = load_image(source_path,mask_path,target_path)
        data = preprocess(image)
        final_image = blend_image(data, BLEND_TYPE, GRAD_MIX)
        save_img = final_image * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite("static/result.jpg", save_img)
        target_path="static/result.jpg"
        mask_path= "Right_eye_mask.png"
        image = load_image_flipped(source_path,mask_path,target_path)
        data = preprocess(image)
        final_image = blend_image(data, BLEND_TYPE, GRAD_MIX)
        save_img = final_image * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite("static/result.jpg", save_img)
        print("Selected image category:", selected_category)
        random_images=[os.path.join("static/nose_clusters",i) for i in os.listdir("static/nose_clusters")]
        return render (request, "single_form.html",{'raw':random_images,'main':"Nose",'next_action':"form3"})
    else:
        # Handle GET requests or render the form again
        # Example:
        return render(request, 'your_template.html')

def select_mouth(request):
    if request.method == 'POST':
        selected_category = request.POST.get('image_category')
        source_path=selected_category
        BLEND_TYPE = 2
        GRAD_MIX = True
        mask_path= "nose_mask.png"
        target_path="static/result.jpg"
        image = load_image(source_path,mask_path,target_path)
        data = preprocess(image)
        final_image = blend_image(data, BLEND_TYPE, GRAD_MIX)
        save_img = final_image * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite("static/result.jpg", save_img)
        random_images=[os.path.join("static/mouth_clusters",i) for i in os.listdir("static/mouth_clusters")]
        return render (request, "single_form.html",{'raw':random_images,'main':"Mouth",'next_action':"form4"})
    else:
        return render(request, 'your_template.html')

def final(request):
    if request.method == 'POST':
        selected_category = request.POST.get('image_category')
        source_path=selected_category
        BLEND_TYPE = 2
        GRAD_MIX = True
        mask_path= "mouth_mask.png"
        target_path="static/result.jpg"
        image = load_image(source_path,mask_path,target_path)
        data = preprocess(image)
        final_image = blend_image(data, BLEND_TYPE, GRAD_MIX)
        save_img = final_image * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite("static/result.jpg", save_img)        
        random_images=["static/result.jpg"]
        return render (request, "single_form.html",{'raw':random_images,'next_action':"final"})
    else:
        return render(request, 'your_template.html')