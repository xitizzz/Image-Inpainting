from efros import efros_algorithm
import settings as st
import time


def main():
    img_names = ["test_im3_board_r.bmp", "test_im2.bmp"]
    for img in img_names:
        print img
        st.set_window(11)
        start = time.time()
        efros_algorithm(str(st.input_path+img), 11, img)
        end = time.time()
        print "Finished in "+str(end-start)+" seconds"

if __name__ == "__main__":
    main()

