import numpy as np
import settings as st
from skimage import io, morphology, exposure, transform, color
from skimage.measure import label, regionprops
from random import randint
from math import ceil, floor
import matplotlib.pyplot as plt
import time

from utilityFunctions import gauss2D, sliding_window, sliding_window_object_removal
from findMatches import find_mathces, find_matches_efficient


def efros_algorithm(src_img_path, window_size, output_name):
    start_time = time.time()
    max_error_threshold = st.ERROR_THRESHOLD
    src_img = io.imread(src_img_path)
    new_image_row = src_img.shape[0]
    new_image_col = src_img.shape[1]
    src_img = color.rgb2gray(src_img)
    is_filled = np.ceil(src_img)
    number_filled = np.sum(is_filled)

    fill_label = label(is_filled)
    plt.imshow(fill_label)
    plt.show()

    # segments = regionprops(fill_label)
    #
    # samples =[]
    #
    # for s in segments:
    #     min_row, min_col, max_row, max_col = s.bbox
    #     samples.append(np.array(src_img[min_row:max_row, min_col:max_col]))
    #
    # # for s in samples:
    # #     plt.imshow(s, cmap="gray")
    # #     plt.show()
    #
    # samples_resized = []
    # for s in samples:
    #     if st.do_resize:
    #         factor = np.sqrt((s.shape[0]*s.shape[1]*len(samples))/10000)
    #     else:
    #         factor = 1
    #     samples_resized.append(transform.resize(s, (ceil(s.shape[0]/factor), (ceil(s.shape[1]/factor)))))



    # for s in samples_resized:
    #     plt.imshow(s, cmap="gray")
    #     plt.show()


    # plt.imshow(src_img, cmap="gray")
    # plt.show()

    # img_row, img_col = np.shape(src_img)

    sigma = st.window_size / 6.4;

    patch_matrix = sliding_window_object_removal(src_img, is_filled)
    print patch_matrix.shape
    number_pixel = src_img.shape[0] * src_img.shape[1]
    interval = ceil(number_pixel/100)
    new_image = src_img

    # #Seed
    # seed_size = st.SEED_SIZE
    # random_row = randint(0, img_row + 1 - seed_size)
    # random_col = randint(0, img_col + 1 - seed_size)
    # seed = src_img[random_row:random_row + seed_size, random_col:random_col + seed_size]
    # new_image[ceil(new_image_row / 2):ceil(new_image_row / 2) + seed_size,
    # ceil(new_image_col / 2):ceil(new_image_col / 2) + seed_size] = seed
    #
    #
    # is_filled = np.zeros((new_image_row, new_image_col))
    # is_filled[ceil(new_image_row / 2):ceil(new_image_row / 2) + seed_size,
    # ceil(new_image_col / 2):ceil(new_image_col / 2) + seed_size] = np.ones((seed_size, seed_size))

    gaussian_mask = gauss2D((window_size, window_size), sigma=sigma)
    print gaussian_mask.shape

    new_image_padded = np.lib.pad(new_image, st.half_window, 'constant', constant_values=0)
    is_filled_padded = np.lib.pad(is_filled, st.half_window, 'constant', constant_values=0)

    while number_filled < number_pixel:
        progress = False
        candidate_pixel_row, candidate_pixel_col = np.nonzero(morphology.binary_dilation(is_filled) - is_filled)
        neighborHood = []
        for i in range(len(candidate_pixel_row)):
            pixel_row = candidate_pixel_row[i]
            pixel_col = candidate_pixel_col[i]
            neighborHood.append(np.sum(is_filled[pixel_row - st.half_window : pixel_row + st.half_window+1,
                                       pixel_col - st.half_window : pixel_col + st.half_window+1]))
        # print candidate_pixel_row.shape
        order = np.argsort(-np.array(neighborHood, dtype=int))
        # print order
        for x, i in enumerate(order): #range(len(candidate_pixel_row)):
            pixel_row = candidate_pixel_row[i]
            pixel_col = candidate_pixel_col[i]
            # if x>0:
            #     if neighborHood[i]>neighborHood[last_i]:
            #         print neighborHood[i]-neighborHood[last_i]
            # last_i = i
            # print pixel_row, pixel_col
            # best_match = find_mathces(new_image_padded[pixel_row - st.half_window + st.half_window:pixel_row + st.half_window +st.half_window+1,
            #                                pixel_col - st.half_window + st.half_window :pixel_col + st.half_window+st.half_window+1],
            #                                src_img,
            #                                is_filled_padded[pixel_row - st.half_window + st.half_window :pixel_row + st.half_window + st.half_window+1,
            #                                pixel_col - st.half_window + st.half_window :pixel_col + st.half_window + st.half_window + 1],
            #                                gaussian_mask)
            best_match = find_matches_efficient(
                new_image_padded[pixel_row - st.half_window + st.half_window:pixel_row + st.half_window + st.half_window + 1,
                pixel_col - st.half_window + st.half_window:pixel_col + st.half_window + st.half_window + 1],
                patch_matrix,
                is_filled_padded[pixel_row - st.half_window + st.half_window:pixel_row + st.half_window + st.half_window + 1,
                pixel_col - st.half_window + st.half_window:pixel_col + st.half_window + st.half_window + 1],
                gaussian_mask)
            """
            DEBUG code to verify correctness of efficient method
            """
            # if len(best_match)!=len(best_match_t):
            #     print "ooooh NO"
            # for i in range(len(best_match)):
            #     if not np.isclose(best_match[i][0], best_match_t[i][0]):
            #         print "Ooops "+str(best_match[i][0])+" "+str(best_match_t[i][0])
            #     if best_match[i][1]!=best_match_t[i][1]:
            #         print "pixel val"
            # print (len(best_match))
            pick = randint(0, len(best_match)-1)
            if best_match[pick][0]<=max_error_threshold:
                new_image_padded[st.half_window+pixel_row][st.half_window+pixel_col] = best_match[pick][1]
                new_image[pixel_row][pixel_col]=best_match[pick][1]
                is_filled_padded[st.half_window+pixel_row][st.half_window+pixel_col] = 1
                is_filled[pixel_row][pixel_col]=1
                number_filled+=1
                if number_filled % interval == 0:
                    if randint(0, 50) == 0:
                        st.quote()
                    else:
                        print "{:d}% --> Time = {:3.2f} sec".format(int(number_filled/interval), time.time()-start_time)
                progress = True
        if not progress:
            max_error_threshold *= 1.1
            print "new threshold = " + str(max_error_threshold)

    # new_image = new_image*255

    io.imsave(st.output_path+output_name, new_image)
    # io.imshow(new_image)
    plt.show()