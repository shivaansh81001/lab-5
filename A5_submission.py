import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from scipy.fftpack import dct
from scipy.fftpack import idct

from A5_utils import zigzag
from A5_utils import inverse_zigzag


# TODO: Implement the function below
def convert_rgb_to_ycbcr(rgb_img):
    """
    Transform the RGB image to a YCbCr image
    """
    #print("rgb",rgb_img)
    
    Ycbcr=[]
    mat = np.array([[0.299000,0.587000,0.114000],
                    [-0.168736,-0.331264,0.500002],
                    [0.500000,-0.418688,-0.081312]])
    
    for row in rgb_img:
        temp_row=[]
        for pixel in row:
            ycbcr_pixel=[]

            for row_in_mat in mat:
                #print(np.sum(row_in_mat*pixel))
                ycbcr_pixel.append(np.sum(row_in_mat*pixel))
            temp_row.append(ycbcr_pixel)  
            #print(temp_row)
        Ycbcr.append(temp_row)

    Ycbcr = np.array(Ycbcr)
    
    Ycbcr[:,:,1]+=128
    Ycbcr[:,:,2]+=128
    Ycbcr = np.uint8(np.round(Ycbcr))
    print("Ycbcr",Ycbcr.shape)
    
    img = Ycbcr
    
    return img

# TODO: Implement the function below
def convert_ycbcr_to_rgb(ycbcr_img):
    """
    Transform the YCbCr image to a RGB image
    """
    ycbcr_img[:,:,1]-=128
    ycbcr_img[:,:,2]-=128

    rgb=[]
    mat = np.array([[1.0, 0.0, 1.40210],
                    [1.0,-0.34414,-0.71414],
                    [1.0,1.77180,0.0]])
    
    for row in ycbcr_img:
        temp_row=[]
        for pixel in row:
            rgb_pixel=[]

            for row_in_mat in mat:
                #print(np.sum(row_in_mat*pixel))
                rgb_pixel.append(np.sum(row_in_mat*pixel))
            temp_row.append(rgb_pixel)  
            #print(temp_row)
        rgb.append(temp_row)

    rgb = np.array(rgb)
    
    rgb = np.uint8(np.round(rgb))
    print("Ycbcr",rgb.shape)
    
    img = rgb

    return img

# TODO: Implement the function below
def dct2D(input_img):
    """
    Function to compute 2D Discrete Cosine Transform (DCT)
    """
    # Apply DCT with type 2 and 'ortho' norm parameters
    result = dct(input_img,type=2,norm="ortho")

    return result

# TODO: Implement the function below
def idct2D(input_dct):
    """
    Function to compute 2D Inverse Discrete Cosine Transform (IDCT)
    """
    # Apply IDCT with type 2 and 'ortho' norm parameters
    result = idct(input_dct,type=2,norm="ortho")

    return result


def part1_encoder():
    """
    JPEG Encoding
    """

    # NOTE: Defining block size
    block_size = 8

    # TODO: Read image using skimage.io
    ###### Your code here ######
    img = io.imread("bird.jpg")

    plt.imshow(img)
    plt.title('input image (RGB)')
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from RGB space to YCbCr space
    img = convert_rgb_to_ycbcr(img)
    
    plt.imshow(np.uint8(img))
    plt.title('input image (YCbCr)')
    plt.axis('off')
    plt.show()

    # TODO: Get size of input image (h, w, c)
    ###### Your code here ######
    h, w, c = img.shape

    # TODO: Compute number of blocks (of size 8-by-8), cast the numbers to int

    nbh = h//block_size ###### Your code here ###### # (number of blocks in height)
    nbw = w//block_size ###### Your code here ###### # (number of blocks in width)

    print(nbh,nbw)

    # TODO: (If necessary) Pad the image, get size of padded image
    H = h+((block_size-(h-(nbh*block_size)))%block_size)###### Your code here ######  # height of padded image
    W = w+((block_size-(w-(nbw*block_size)))%block_size)###### Your code here ######  # width of padded image
    print("new H,W after padding", H,W)
    # TODO: Create a numpy zero matrix with size of H,W,3 called padded img
    padded_img = np.zeros((H,W,3))

    # TODO: Copy the values of img into padded_img[0:h,0:w,:]
    ###### Your code here ######
    for i in range(h):
        for j in range(w):
            padded_img[i,j]=img[i,j]

    # TODO: Display padded image
    plt.imshow(np.uint8(padded_img))
    plt.title('input padded image')
    plt.axis('off')
    plt.show()


    # TODO: Create the quantization matrix
    # Refer to eClass for imformation on filling these matrices
    quantization_matrix_Y = np.array([[16,11,10,16,24,40,51,61],
                                      [12,12,14,19,26,58,60,55],
                                      [14,13,16,24,40,57,69,56],
                                      [14,17,22,29,51,87,80,62],
                                      [18,22,37,56,68,109,103,77],
                                      [49,64,78,87,103,121,120,101],
                                      [72,92,95,98,112,100,103,99]])# quantization table for Y channels
    
    quantization_matrix_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                        [18, 21, 26, 66, 99, 99, 99, 99],
                                        [24, 26, 56, 99, 99, 99, 99, 99],
                                        [47, 66, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99]])# quantization table for Cb and Cr channels
    ###### Your code here ######

    # TODO: Initialize variables for compression calculations (only for the Y channel)
    ###### Your code here ######

    # NOTE: Iterate over blocks
    for i in range(nbh):
        
        # Compute start and end row indices of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size
        
        for j in range(nbw):
            
            # Compute start and end column indices of the block
            col_ind_1 = j * block_size 
            col_ind_2 = col_ind_1 + block_size
            
            # TODO: Select current block to process using calculated indices (through slicing)
            Yblock = ...###### Your code here ######
            Cbblock = ...###### Your code here ######
            Crblock = ...###### Your code here ######
            
            # TODO: Apply dct2d() to selected block             
            YDCT = ...###### Your code here ######
            CbDCT = ...###### Your code here ######
            CrDCT = ...###### Your code here ######

            # TODO: Quantization
            # Divide each element of DCT block by corresponding element in quantization matrix
            quantized_YDCT = ...###### Your code here ######
            quantized_CbDCT = ...###### Your code here ######
            quantized_CrDCT = ...###### Your code here ######

            # TODO: Reorder DCT coefficients into block (use zigzag function)
            reordered_Y = ...###### Your code here ######
            reordered_Cb = ...###### Your code here ######
            reordered_Cr = ...###### Your code here ######

            # TODO: Reshape reordered array to 8-by-8 2D block
            reshaped_Y = ...###### Your code here ######
            reshaped_Cb = ...###### Your code here ######
            reshaped_Cr = ...###### Your code here ######

            # TODO: Copy reshaped matrix into padded_img on current block corresponding indices
            ###### Your code here ######

            # TODO: Compute pixel locations with non-zero values before and after quantization (only in Y channel)
            # TODO: Compute total number of pixels
            ###### Your code here ####

    plt.imshow(np.uint8(padded_img))
    plt.title('encoded image')
    plt.axis('off')
    plt.show()

    # TODO: Calculate percentage of pixel locations with non-zero values before and after to measure degree of compression 
    before_compression = ... ###### Your code here ####
    after_compression = ... ###### Your code here ####

    # Print statements as shown in eClass
    print('Percentage of non-zero elements in Luma channel:')
    print('Before compression: ', before_compression, '%')
    print('After compression: ', after_compression, '%')


    # Writing h, w, c, block_size into a .txt file
    np.savetxt('size.txt', [h, w, c, block_size])

    # Writing the encoded image into a file
    np.save('encoded.npy', padded_img)


def part2_decoder():
    """
    JPEG Decoding
    """

    # TODO: Load 'encoded.npy' into padded_img (using np.load() function)
    ###### Your code here ######
    padded_img = ...

    # TODO: Load h, w, c, block_size and padded_img from the size.txt file
    ###### Your code here ######
    h, w, c, block_size = ...

    # TODO: 6. Get size of padded_img, cast to int if needed
    ###### Your code here ######

    # TODO: Create the quantization matrix (Same as before)
    quantization_matrix_Y = ... # quantization table for the Y channel
    quantization_matrix_CbCr = ... # quantization table for the Y channel
    
    ###### Your code here ######

    # TODO: Compute number of blocks (of size 8-by-8), cast to int
    nbh = ... ###### Your code here ###### # (number of blocks in height)
    nbw = ... ###### Your code here ###### # (number of blocks in width)

    # TODO: iterate over blocks
    for i in range(nbh):
        
            # Compute start and end row indices of the block
            row_ind_1 = i * block_size
            
            row_ind_2 = row_ind_1 + block_size
            
            for j in range(nbw):
                
                # Compute start and end column indices of the block
                col_ind_1 = j * block_size

                col_ind_2 = col_ind_1 + block_size
                
                # TODO: Select current block to process using calculated indices
                Yblock = ... ###### Your code here ######
                Cbblock = ... ###### Your code here ######
                Crblock = ... ###### Your code here ######
                
                # TODO: Reshape 8-by-8 2D block to 1D array
                Yreshaped = ... ###### Your code here ######
                Cbreshaped = ... ###### Your code here ######
                Crreshaped = ... ###### Your code here ######
                
                # TODO: Reorder array into block (use inverse_zigzag function)
                Yreordered = ... ###### Your code here ######
                Cbreordered = ... ###### Your code here ######
                Crreordered = ... ###### Your code here ######
                
                # TODO: De-quantization
                # Multiply each element of reordered block by corresponding element in quantization matrix
                dequantized_YDCT = ... ###### Your code here ######
                dequantized_CbDCT = ... ###### Your code here ######
                dequantized_CrDCT = ... ###### Your code here ######
                
                # TODO: Apply idct2d() to reordered matrix 
                YIDCT = ... ###### Your code here ######
                CbIDCT = ... ###### Your code here ######
                CrIDCT = ... ###### Your code here ######

                # TODO: Copy IDCT matrix into padded_img on current block corresponding indices
                ###### Your code here ######

    # TODO: Remove out-of-range values
    ###### Your code here ######

    plt.imshow(np.uint8(padded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()

    # TODO: Get original sized image from padded_img
    ###### Your code here ######
    decoded_img = ...

    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from YCbCr to RGB
    decoded_img = convert_ycbcr_to_rgb(decoded_img)
    
    # TODO: Remove out-of-range values
    ###### Your code here ######
    
    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded image (RGB)')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    part1_encoder()
    part2_decoder()

