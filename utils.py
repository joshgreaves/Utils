import cv2
import numpy as np
import os


def batch(data, batch_size=32):
    """Create a new batch of data
    Uses the size of the first element in the list to get the max index
    
    Parameters
    ----------
    data : list(numpy.ndarray) or numpy.ndarray
        A list of the data to batch, or one array of data
    batch_size : int
        The size of the batch to generate

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        An subset of data, and a subset of labels
    """
    results = None
    if type(data) == list:
        indices = np.random.choice(len(data[0]), batch_size, False)
        results = []
        for x in data:
            results.append(x[indices])

    elif type(data) == np.ndarray:
        indices = np.random.choice(len(data), batch_size, False)
        results = data[indices]

    else:
        raise RuntimeError("Invalid type for data, must be a list or numpy array")
    return results




def load_image(path, color=True, resize=False,
              height=100, width=100, maintain_aspect_ratio=True,
              crop=False, padding="White"):
    """Loads a single image
    If resize and crop, it will be cropped
    If resize and not crop and maintain_aspect_ratio it will be padded
    If resize and not maintain_aspect_ration it will be squashed/stretched

    Parameters
    ----------
    path : str
        The full or relative path of the image
    color : bool
        True for loading the image in color, or False for grayscale
    resize : bool
        True to resize the images when loading them
    height : int
        The height to load the picture at (only if resize == True)
    width : int
        The width to load the picture at (only if resize == True)
    maintain_aspect_ratio : bool
        True to maintain aspect ratio (will be padded) (only if resize == True)
    crop : bool
        If True, it will fit the shortest side and crop the rest (only if resize == True)
    padding : str
        "White" or "Black"
    
    Returns
    -------
    numpy.ndarray
        The image in a numpy array
    """
    img = cv2.imread(path, color)
    
    if resize:
        if maintain_aspect_ratio:
            if crop:
                if img.shape[0] / float(height) < img.shape[1] / float(height):
                    img = resize_image(img, height=height)
                else:
                    img = resize_image(img, width=width)
            else:
                if img.shape[0] / float(height) < img.shape[1] / float(width):
                    img = resize_image(img, width=width)
                else:
                    img = resize_image(img, height=height)
        else:
            img = resize_image(img, height=height, width=width)
        
        img = crop_or_pad(img, height, width, padding)

    return img



def save_image(img, path):
    """Saves a single image

    Parameters
    ----------
    img : numpy.ndarray
        The image to save
    path : str
        The path and name to save the image to
    
    Returns
    -------
    None

    """
    cv2.imwrite(path, img)


def load_image_names(path):
    """Loads the name of all images in a directory
    Will look for all .jpeg, .jpg, .png, and .tiff files

    Parameters
    ----------
    path : str
        The full or relative path of the directory

    Returns
    -------
    list(str)
        A sorted list of all the file names

    """
    return sorted( img for img in os.listdir(path) if (os.path.isfile(os.path.join(path, img))
                                                    and img.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))))


def load_images_in_directory(path, color=True,
                             height=100, width=100, maintain_aspect_ratio=True,
                             crop=False, padding="White"):
    """Loads all images in a directory
    Will look for all .jpeg, .jpg, .png, and .tiff files
    
    Parameters
    ----------
    path : str
        The full or relative path of the image
    color : bool
        True for loading the image in color, or False for grayscale
    height : int
        The height to load the picture at
    width : int
        The width to load the picture at
    maintain_aspect_ratio : bool
        True to maintain aspect ratio (will be padded)
    crop : bool
        If True, it will fit the shortest side and crop the rest
    padding : str
        "White" or "Black"
    
    Returns
    -------
    numpy.ndarray
        An array of all the images of shape [num_images, height, width, channels]
    """

    img_names = load_image_names(path)
    result = np.empty([len(img_names), height, width, 3 if color else 1], dtype=np.uint8)
    for i, x in enumerate(img_names):
        result[i] = load_image(os.path.join(path, x), color=color,
                               resize=True, height=height, width=width,
                               maintain_aspect_ratio=maintain_aspect_ratio, crop=crop,
                               padding=padding)
    return result



def resize_image(img, height=0, width=0):
    """Resize an image to a desired height and width
    
    Parameters
    ----------
    img : numpy.ndarray
        The image to resize
    height : int
        The max height you want the image to be. If 0, it is calculated from Width
    width : int
        The max width you want the image to be. If 0, it is calculated from Height
    
    Returns
    -------
    numpy.ndarray
        The resized image
    """

    # If both are zero, we don't know what to resize it to!
    if (height == 0 and width == 0):
        raise ValueError("Height and Width can't both be 0!")
    elif (height < 0 or width < 0):
        raise ValueError("Height or Width can't be below 0")
    elif (height == 0):
        # We need to caluclate the scale from the width
        scale = float(width) / img.shape[1]
    elif (width == 0):
        # we need to calculate the scale from the height
        scale = float(height) / img.shape[0]
    else:
        # In this case, the image will not maintain aspect ratio
        if img.shape[0] > height:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # If the scale factor is larger:
    if scale > 1:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    



def crop_or_pad(img, height, width, padding="White"):
    """Puts the image into a new array of Height x Width, and crops or pads as necessary

    Parameters
    ----------
    img : numpy.ndarray
        The image to put onto the canvas
    height : int
        The desired height of the returned image
    width : int
        The desired width of the returned image
    padding : str
        The color to pad with. "White" or "Black" 
    
    Returns
    -------
    numpy.ndarray
        The image after having been cropped or padded
    """

    start_y = int(round((height - img.shape[0]) / 2.0))
    start_x = int(round((width - img.shape[1]) / 2.0))
    end_x = 0
    end_y = 0

    # If these are less than 0, we must trim some
    img_start_y = 0
    img_start_x = 0
    img_end_y = img.shape[0]
    img_end_x = img.shape[1]
    
    if start_y < 0:
        img_start_y -= start_y
        img_end_y = img_start_y + height
        start_y = 0
        end_y = height
    else:
        end_y = start_y + img.shape[0]

    if start_x < 0:
        img_start_x -= start_x
        img_end_x = img_start_x + width
        start_x = 0
        end_x = width
    else:
        end_x = start_x + img.shape[1]

    # If it is a full color image
    if img.shape[2] == 3:
        if padding == "White":
            array = np.full((height, width, 3), 255, dtype=np.uint8)
        elif padding == "Black":
            array = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            raise ValueError("Unknown parameter for pading, " + padding + ". Must be Black or White")
    else:
        if padding == "White":
            array = np.full((height, width, 1), 255, dtype=np.uint8)
        elif padding == "Black":
            array = np.zeros((height, width, 1), dtype=np.uint8)
        else:
            raise ValueError("Unknown parameter for padding, " + padding + ". Must be Black or White")
    
    # Insert the image into the array
    array[start_y:end_y, start_x:end_x] = img[img_start_y:img_end_y, img_start_x:img_end_x]
    return array




def display_images(imgs):
    """Displays all images in a batch
    
    Parameters
    ----------
    img : numpy.ndarray
        The image in a numpy array [batch, height, width, channels]
    
    Returns
    -------
    None
    """

    for i, x in enumerate(img):
        cv2.imshow("img" + str(i), x)
    cv2.waitKey()
    cv2.destroyAllWindows()