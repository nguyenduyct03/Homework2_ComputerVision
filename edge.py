import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    kernel = np.flip(np.flip(kernel,0),1)
    ### YOUR CODE HERE
    for y in range(0,Hi):
        for x in range(0,Wi):
            out[y,x] = np.sum(kernel*padded[y:y+Hk,x:x+Wk])
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = (size - 1)/2
    for y in range(0,size):
        for x in range(0,size):
            a = 1/(2*np.pi*(sigma**2))
            b = (y-k)**2
            c = (x-k)**2
            d = (b+c)/2*(sigma**2)
            kernel[y,x] = a*np.exp(-d)
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[1,0,-1]])
    out = conv(img,kernel)/2
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[1],[0],[-1]])
    out = conv(img,kernel)/2
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    Gx = partial_x(img)
    Gy = partial_y(img)

    ### YOUR CODE HERE
    theta = np.arctan2(Gy,Gx)*180/np.pi
    G = np.sqrt(Gx**2+Gy**2)
    for y in range(0,img.shape[0]):
        for x in range (0,img.shape[1]):
            if (theta[y,x] < 0):
                theta[y,x] = 360 + theta[y,x]
    #Chú ý: Cần phải đổi giá trị của theta từ radiant sang độ trước khi trả về
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))
    Gp = np.pad(G, ((1,1),(1,1)), 'constant', constant_values=0)

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    # Tim he so goc cua diem anh
    direc = theta/45
    for y in range(G.shape[0]):
        for x in range(G.shape[1]):
            # Neu he so goc = 0/4 tuong duong goc = 0/180 do
            if (direc[y,x] == 0 or direc[y,x] == 4):
                # Lay ra hang ngang cac diem lan can
                kernel = np.array([ [0,0,0],
                                    [1,1,1],
                                    [0,0,0] ])
                Gt = kernel*Gp[y:y+3,x:x+3]
                # So sanh diem dang xet, tuong duong diem trung tam cua Gt vs gia tri max cua toan Gt
                if (Gt[1,1] == np.max(Gt)):
                    out[y,x] = Gt[1,1]
            # Neu he so goc = 1/5 tuong duong goc = 45/225 do
            if (direc[y,x] == 1 or direc[y,x] == 5):
                # Lay ra hang cheo 45 do cac diem lan can
                kernel = np.array([ [1,0,0],
                                    [0,1,0],
                                    [0,0,1] ])
                Gt = kernel*Gp[y:y+3,x:x+3]
                # So sanh diem dang xet, tuong duong diem trung tam cua Gt vs gia tri max cua toan Gt
                if (Gt[1,1] == np.max(Gt)):
                    out[y,x] = Gt[1,1]
            # Neu he so goc = 2/6 tuong duong goc = 90/270 do
            if (direc[y,x] == 2 or direc[y,x] == 6):
                # Lay ra hang doc cac diem lan can
                kernel = np.array([ [0,1,0],
                                    [0,1,0],
                                    [0,1,0] ])
                Gt = kernel*Gp[y:y+3,x:x+3]
                # So sanh diem dang xet, tuong duong diem trung tam cua Gt vs gia tri max cua toan Gt
                if (Gt[1,1] == np.max(Gt)):
                    out[y,x] = Gt[1,1]
            # Neu he so goc = 3/7 tuong duong goc = 135/315 do
            if (direc[y,x] == 3 or direc[y,x] == 7):
                # Lay ra hang cheo 135 do cac diem lan can
                kernel = np.array([ [0,0,1],
                                    [0,1,0],
                                    [1,0,0] ])
                Gt = kernel*Gp[y:y+3,x:x+3]
                # So sanh diem dang xet, tuong duong diem trung tam cua Gt vs gia tri max cua toan Gt
                if (Gt[1,1] == np.max(Gt)):
                    out[y,x] = Gt[1,1]
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    strong_edges = img > high
    weak_edges = np.logical_and((img <= high),(img > low))
    ### END YOUR CODE
    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W), binary image
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    while indices.shape[0] != 0:
        y, x = indices[0]
        indices = np.delete(indices,0,0)
        edges[y,x] = 1  
        nlist = get_neighbors(y,x,H,W)
        while (len(nlist) != 0):
            i, j = nlist.pop()
            if (weak_edges[i,j] and edges[i,j] == 0):
                indices = np.append(indices,[[i,j]],0)
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: Grayscale image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W), binary image
    """
    ### YOUR CODE HERE

    kernel = gaussian_kernel(kernel_size,sigma)
    smoothed = conv(img,kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G,theta)
    strong_edges, weak_edges = double_thresholding(nms,high,low)
    edge = link_edges(strong_edges, weak_edges)

    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for y, x in zip(ys,xs):
            for t in range(num_thetas):
                r = int(x*cos_t[t] + y*sin_t[t])
                accumulator[r+diag_len,t] += 1            
    ### END YOUR CODE

    return accumulator, rhos, thetas
